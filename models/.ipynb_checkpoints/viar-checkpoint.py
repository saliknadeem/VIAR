#!/usr/bin/env python3
"""
  View-Invariant Action Representations, referenced from
J. Li, Y. Wong, Q. Zhao, and M. S. Kankanhalli, “Unsupervised Learning of 
View-invariant Action Representations.," NeurIPS, 2018.
"""
import os
import sys
import numpy as np
import math
import json
import time
import argparse
import operator
from functools import reduce

import torch
import torch.nn as nn

from networks import CNN, Encoder, CrossViewDecoder, \
                     ReconstructionDecoder, ViewClassifier, ActionClassifier, MultiTaskLossWrapper

sys.path.append('..')
#from dataloader.NTURGBDwithFlowLoader import NTURGBDwithFlowLoader
from dataloader.NTURGBDwithFlowLoaderVal import NTURGBDwithFlowLoader, NTURGBDwithFlowLoaderValidation
from utils.utils import setCheckpointFileDict, testIters, trainIters

RGB_INPUT_SHAPE = (3,224,224)
DEPTH_INPUT_SHAPE = (1,224,224)
RGBD_INPUT_SHAPE = (4,224,224)
FLOW_SHAPE = (3,224,224)
ALL_MODELS = ['encodercnn', 'encoder','crossviewdecoder','crossviewdecodercnn',
              'reconstructiondecoder','viewclassifier', 'actionclassifier', 'multitasklosswrapper']
LOG_PREFIX = 'VIAR'

def main():
  args = get_args()

  margs = {}
  margs['json_file'] = os.path.join(args.ntu_dir, 'ntu_rgbd_videonames.min.json')
  margs['label_file'] = os.path.join(args.ntu_dir, 'ntu_rgbd_action_labels.txt')
  margs['flow_h5_dir'] = os.path.join(args.ntu_dir, 'Extracted3DFlowH5')
  margs['rgb_h5_dir'] = os.path.join(args.ntu_dir, 'nturgb+d_rgb_pngs_320x240_lanczos_h5')
  margs['depth_h5_dir'] = os.path.join(args.ntu_dir, 'MaskedDepthMaps_320x240_h5')

  # Use cuda device if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  margs['device'] = device

  checkpoint_files = setCheckpointFileDict(ALL_MODELS, args.checkpoint_files)

  models = build_models(args, device=device)

  if args.for_what == 'train':
    main_train(checkpoint_files, args, margs, models)

  elif args.for_what == 'test':
    main_test(checkpoint_files, args, margs, models)
  elif args.for_what == 'trainAction':
    main_trainAction(checkpoint_files, args, margs, models)

  else:
    raise NotImplementedError(
      'Given "{}" mode is not implemented'.format(args.for_what)
      )

def build_models(args, device='cuda'):
  models = {}
  from torchsummary import summary

  if args.modality == 'rgb':
      models['encodercnn'] = CNN(
        input_shape=RGB_INPUT_SHAPE, model_name=args.encoder_cnn_model,input_channel=3).to(device)
  elif args.modality == 'depth':
      models['encodercnn'] = CNN(
        input_shape=DEPTH_INPUT_SHAPE, model_name=args.encoder_cnn_model,input_channel=1).to(device)  
  elif args.modality == 'rgbd':
      models['encodercnn'] = CNN(
        input_shape=RGBD_INPUT_SHAPE, model_name=args.encoder_cnn_model,input_channel=4).to(device) 
  elif args.modality == 'pdflow':
      models['encodercnn'] = CNN(
        input_shape=FLOW_SHAPE, model_name=args.encoder_cnn_model,input_channel=3).to(device) 
  
  
  ##################print(summary(models['encodercnn'], (1, 224, 224)))

  models['encoder'] = Encoder(
    input_shape=models['encodercnn'].out_size, encoder_block='convbilstm', #encoder_block='convbilstm',  #skl
    hidden_size=args.encoder_hid_size).to(device)


  models['crossviewdecodercnn'] = CNN(
    input_shape=DEPTH_INPUT_SHAPE, model_name=args.encoder_cnn_model, 
    input_channel=1).to(device)

  crossviewdecoder_in_size = list(models['crossviewdecodercnn'].out_size)
  crossviewdecoder_in_size[0] = crossviewdecoder_in_size[0] * 3
  crossviewdecoder_in_size = torch.Size(crossviewdecoder_in_size)
  models['crossviewdecoder'] = CrossViewDecoder(
    input_shape=crossviewdecoder_in_size).to(device)

  models['reconstructiondecoder'] = ReconstructionDecoder(
    input_shape=models['encoder'].out_size[1:]).to(device)

  models['viewclassifier'] = ViewClassifier(
    input_size=reduce(operator.mul, models['encoder'].out_size[1:]), 
    num_classes=5,
    reverse=(not args.disable_grl)).to(device)

  models['actionclassifier'] = ActionClassifier(
    input_size=reduce(operator.mul, models['encoder'].out_size[1:]), 
    num_classes=60).to(device)

  models['multitasklosswrapper'] = MultiTaskLossWrapper(2,models).to(device)

  return models

def main_train(checkpoint_files, args, margs, models):
  train_loader, train_dataset = NTURGBDwithFlowLoader(
    json_file=margs['json_file'], 
    label_file=margs['label_file'], 
    rgb_h5_dir=margs['rgb_h5_dir'], 
    depth_h5_dir=margs['depth_h5_dir'], 
    flow_h5_dir=margs['flow_h5_dir'], 
    target_length=args.target_length, 
    subset='train', 
    visual_transform=args.visual_transform, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=args.num_workers, 
    pin_memory=True
    ) 

  val_loader, val_dataset = NTURGBDwithFlowLoader(
    json_file=margs['json_file'], 
    label_file=margs['label_file'], 
    rgb_h5_dir=margs['rgb_h5_dir'], 
    depth_h5_dir=margs['depth_h5_dir'], 
    flow_h5_dir=margs['flow_h5_dir'], 
    target_length=args.target_length, 
    subset='test', 
    visual_transform=args.visual_transform, 
    batch_size= args.batch_size, 
    shuffle=False, 
    num_workers=args.num_workers, 
    pin_memory=True
    ) 
  #exit()
  trainIters(run, args.target_modules, 
             train_loader, train_dataset, val_loader, val_dataset,
             models=models, all_models=ALL_MODELS, log_prefix=LOG_PREFIX, 
             checkpoint_files=checkpoint_files, save_dir=args.save_dir, 
             args=args, device=margs['device'])


def main_trainAction(checkpoint_files, args, margs, models):
  train_loader, train_dataset = NTURGBDwithFlowLoader(
    json_file=margs['json_file'], 
    label_file=margs['label_file'], 
    rgb_h5_dir=margs['rgb_h5_dir'], 
    depth_h5_dir=margs['depth_h5_dir'], 
    flow_h5_dir=margs['flow_h5_dir'], 
    target_length=args.target_length, 
    subset='train', 
    visual_transform=args.visual_transform, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=args.num_workers, 
    pin_memory=True
    ) 

  val_loader, val_dataset = NTURGBDwithFlowLoaderValidation(
    json_file=margs['json_file'], 
    label_file=margs['label_file'], 
    rgb_h5_dir=margs['rgb_h5_dir'], 
    depth_h5_dir=margs['depth_h5_dir'], 
    flow_h5_dir=margs['flow_h5_dir'], 
    target_length=args.target_length, 
    subset='test', 
    visual_transform=args.visual_transform, 
    batch_size=1,  #args.batch_size, 
    shuffle=False, 
    num_workers=args.num_workers, 
    pin_memory=True
    )
    

  trainIters(runAction, args.target_modules, 
             train_loader, train_dataset, val_loader, val_dataset,
             models=models, all_models=ALL_MODELS, log_prefix=LOG_PREFIX, 
             checkpoint_files=checkpoint_files, save_dir=args.save_dir, 
             args=args, device=margs['device'])





def main_test(checkpoint_files, args, margs, models):
  test_loader, test_dataset = NTURGBDwithFlowLoader(
    json_file=margs['json_file'], 
    label_file=margs['label_file'], 
    rgb_h5_dir=margs['rgb_h5_dir'], 
    depth_h5_dir=margs['depth_h5_dir'], 
    flow_h5_dir=margs['flow_h5_dir'], 
    target_length=args.target_length, 
    subset='test', 
    visual_transform=args.visual_transform, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=args.num_workers, 
    pin_memory=True
    ) 

  testIters(run, test_loader, test_dataset, models=models, 
            checkpoint_files=checkpoint_files, args=args, device=margs['device'])

def run(split, sample, models, target_modules=[], device='cuda',
        optimizers=None, criterions=None, args=None):
  result = {}
  result['logs'] = {}
  result['output'] = {}
  if split == 'train':
    set_grad = True
    for m in models:
      if m in target_modules:
        models[m].train()
        optimizers[m].zero_grad()
      else:
        models[m].eval()
  else:
    set_grad = False
    for m in models:
      models[m].eval()

  batch_size = len(sample['videoname'])
  target_length = len(sample['rgbs'][0])

  # Encoder
  if args.modality == 'rgb':
      rgb_input = sample['rgbs'].view(
        (batch_size*target_length,) + RGB_INPUT_SHAPE
        ).to(device)
      encodercnn_output = models['encodercnn'](rgb_input)
  elif args.modality == 'depth':
      depth_input = sample['depths'].view(
        (batch_size*target_length,) + DEPTH_INPUT_SHAPE
        ).to(device)
      encodercnn_output = models['encodercnn'](depth_input)
  elif args.modality == 'rgbd':
      #print("sample['depths']===",sample['depths'].shape)
      #print("sample['rgbs']===",sample['rgbs'].shape)
      rgbd = torch.cat((sample['rgbs'], sample['depths']), 2)
      #print("combined===",rgbd.shape)        
      #exit()
      rgbd_input = rgbd.view(
        (batch_size*target_length,) + RGBD_INPUT_SHAPE
        ).to(device)
      encodercnn_output = models['encodercnn'](rgbd_input)
  elif args.modality == 'pdflow':
      #print("-----=-=-=-sample['flows']",sample['flows'].shape)
      pdflow_input = torch.Tensor(np.pad(sample['flows'],[(0, 0), (0, 0),(0, 0), (98, 98),(98, 98)],mode='constant'))
      #print("-----=-=-=-pdflow_input",pdflow_input.shape)
      pdflow_input = pdflow_input.view(
        (batch_size*target_length,) + (3,224,224)
        ).to(device)
      #print("-----=-=-=-pdflow_input",pdflow_input.shape)
      encodercnn_output = models['encodercnn'](pdflow_input)
  
  
  #print("-------pdflow_input=",pdflow_input.shape)
  #print("-------encoderCNNout=",encodercnn_output.shape)
  #print("-------encoderCNNout elem=",encodercnn_output.numel())
  #print("-------models['encodercnn'].out_size=",models['encodercnn'].out_size)

  encodercnn_output = encodercnn_output.view(
    (batch_size, target_length) + models['encodercnn'].out_size )
  #print("-------encoderCNNout=",encodercnn_output.shape)
  encoder_output, _ = models['encoder'](encodercnn_output) # (batch, seq_len, c, h, w)
  
  #print("-------encoderout=",encoder_output.shape)
    
  #print("-------encoderout=",encoder_output.contiguous().view(
  #  (batch_size*target_length,) + models['encoder'].out_size[1:] ).shape)
   
  #exit()
  if split == 'test':
    result['output']['encoder_output'] = encoder_output
  encoder_output = encoder_output.contiguous().view(
    (batch_size*target_length,) + models['encoder'].out_size[1:] )




  # CrossViewDecoder
  otherview_depth_input = sample['otherview_depths'].view(
    (batch_size*target_length,) + DEPTH_INPUT_SHAPE ).to(device)
  otherview2_depth_input = sample['otherview2_depths'].view(
    (batch_size*target_length,) + DEPTH_INPUT_SHAPE ).to(device) #skl

  crossviewcnn_output = models['crossviewdecodercnn'](otherview_depth_input)
  crossviewcnn_output2 = models['crossviewdecodercnn'](otherview2_depth_input) #skl
  #print('crossviewcnn_output======',crossviewcnn_output.shape)
  #print('encoder_output======',encoder_output.shape)



  #print('encodercnn_output======',encodercnn_output.shape)
  #print('encoder_output======',encoder_output.shape)
  #print('crossviewcnn_output======',crossviewcnn_output.shape) 
  #encodercnn_output = np.pad(encodercnn_output.cpu,[(0, 0),(0, 0),(0, 0), (98, 98),(98, 98)],mode='constant')
  #print('encodercnn_output======',encodercnn_output.shape)

  crossview_output = models['crossviewdecoder'](crossviewcnn_output, encoder_output)
  crossview_output2 = models['crossviewdecoder'](crossviewcnn_output2, encoder_output) #skl
  crossview_output = crossview_output.view(
    (batch_size, target_length) + models['crossviewdecoder'].out_size )
  crossview_output2 = crossview_output2.view(
    (batch_size, target_length) + models['crossviewdecoder'].out_size ) #skl
  #print('crossviewcnn_output======',crossviewcnn_output.shape)  
  #exit()

  # ReconstructionDecoder
  reconstruct_output = models['reconstructiondecoder'](encoder_output)
  reconstruct_output = reconstruct_output.view(
    (batch_size, target_length) + models['reconstructiondecoder'].out_size )

    
  # ViewClassifier
  viewclassify_output = models['viewclassifier'](
    encoder_output.view(batch_size*target_length,-1) )
  viewclassify_output = viewclassify_output.view(
    (batch_size, target_length) + (models['viewclassifier'].num_classes,) )  

    
    
  ######################## MTL #################################  
  #loss, log_vars = models['multitasklosswrapper'](sample, criterions, encoder_output)
  
  if split in ['train', 'validate']:
    if set_grad:
      if 'encodercnn' in target_modules: optimizers['encodercnn'].zero_grad()
      if 'encoder' in target_modules: optimizers['encoder'].zero_grad()
      if 'crossviewdecodercnn' in target_modules: optimizers['crossviewdecodercnn'].zero_grad()
      if 'crossviewdecoder' in target_modules: optimizers['crossviewdecoder'].zero_grad()
      if 'reconstructiondecoder' in target_modules: optimizers['reconstructiondecoder'].zero_grad()
      if 'viewclassifier' in target_modules: optimizers['viewclassifier'].zero_grad() #actionaction needed
      if 'multitasklosswrapper' in target_modules: optimizers['multitasklosswrapper'].zero_grad() #actionaction needed
                
    

    

    total_loss = 0
    #view_accuracy = 0 #skl
    correct=0
    crossview_loss = 0
    
    
    #print('======crossview_output=====',crossview_output.shape)
    #print('======sample[otherview_flows]=====',sample['otherview_flows'].shape)    
    
    crossview_loss1 = criterions['crossview'](crossview_output, sample['otherview_flows'].to(device))
    crossview_loss2 = criterions['crossview'](crossview_output2, sample['otherview2_flows'].to(device))
    ####crossview_loss = criterions['crossview'](crossview_output, sample['otherview_flows'].to(device))
    crossview_loss = crossview_loss1+crossview_loss2

    reconstruct_loss = criterions['reconstruct'](reconstruct_output, sample['flows'].to(device))
    viewclassify_loss = criterions['viewclassify'](viewclassify_output, sample['view_id'].long().to(device))
    #total_loss += (crossview_loss +  loss )   #   0.5 * reconstruct_loss + 0.05 * viewclassify_loss)
    total_loss += (crossview_loss +   0.5 * reconstruct_loss + 0.05 * viewclassify_loss)

    
    
    
    
    """ Code to see accuracy """
    """
    accuracy_out = criterions['view_accuracy'](viewclassify_output.to(device))
    viewclassify_max_idx = torch.argmax(accuracy_out, 2, keepdim=False)
    view_IDs = torch.argmax(sample['view_id'], -1, keepdim=False)
    true_preds = [view_IDs[i].repeat(viewclassify_max_idx.shape[-1]) for i in range(viewclassify_max_idx.shape[0])]
    correct +=  (torch.sum(  torch.eq(viewclassify_max_idx, torch.stack(true_preds).to(device))   ,dim=-1)/float(target_length)).mean()
    view_accuracy += 100. * correct
    """
    
    if set_grad: #and total_loss != 0:
      total_loss.backward()
      #loss.backward(retain_graph=True)
      #crossview_loss.backward()
      if 'encodercnn' in target_modules: optimizers['encodercnn'].step()
      if 'encoder' in target_modules: optimizers['encoder'].step()
      if 'crossviewdecodercnn' in target_modules: optimizers['crossviewdecodercnn'].step()
      if 'crossviewdecoder' in target_modules: optimizers['crossviewdecoder'].step()
      if 'reconstructiondecoder' in target_modules: optimizers['reconstructiondecoder'].step()
      if 'viewclassifier' in target_modules: optimizers['viewclassifier'].step()
      if 'multitasklosswrapper' in target_modules: optimizers['multitasklosswrapper'].step()        

    result['logs']['loss'] = total_loss.item() #if total_loss > 0 else 0
    #result['logs']['loss'] = loss.item() if loss > 0 else 0
    result['logs']['crossview_loss'] = crossview_loss.item() #if crossview_loss > 0 else 0
    result['logs']['reconstruct_loss'] = reconstruct_loss.item() #if reconstruct_loss > 0 else 0
    result['logs']['viewclassify_loss'] = viewclassify_loss.item() #if viewclassify_loss > 0 else 0
    #result['logs']['viewclassify_accuracy'] = view_accuracy.item() if view_accuracy > 0 else 0

  return result


def runAction(split, sample, models, target_modules=[], device='cuda',
        optimizers=None, criterions=None, args=None):
  result = {}
  result['logs'] = {}
  result['output'] = {}
  if split == 'train':
    set_grad = True
    batch_size = len(sample['videoname'])
    target_length = len(sample['rgbs'][0])
    for m in models:
      if m in target_modules:
        models[m].train()
        optimizers[m].zero_grad()
      else:
        models[m].eval()
  else:
    set_grad = False
    batch_size = 10 #len(sample['videoname'])
    target_length = 6
    for m in models:
      models[m].eval()

  #print("----------------args.modality",args.modality)
  # Encoder
  if args.modality == 'rgb':
      rgb_input = sample['rgbs'].view(
        (batch_size*target_length,) + RGB_INPUT_SHAPE
        ).to(device)
      encodercnn_output = models['encodercnn'](rgb_input)
  elif args.modality == 'depth':
      depth_input = sample['depths'].view(
        (batch_size*target_length,) + DEPTH_INPUT_SHAPE
        ).to(device)
      encodercnn_output = models['encodercnn'](depth_input)
  elif args.modality == 'rgbd':
      #print("sample['depths']===",sample['depths'].shape)
      #print("sample['rgbs']===",sample['rgbs'].shape)
      rgbd = torch.cat((sample['rgbs'], sample['depths']), 2)
      #print("combined===",rgbd.shape)        
      #exit()
      rgbd_input = rgbd.view(
        (batch_size*target_length,) + RGBD_INPUT_SHAPE
        ).to(device)
      encodercnn_output = models['encodercnn'](rgbd_input)
  elif args.modality == 'pdflow': 
      #print("-----=-=-=-sample['flows']",sample['flows'].shape)
      pdflow_input = torch.Tensor(np.pad(sample['flows'],[(0, 0), (0, 0),(0, 0), (98, 98),(98, 98)],mode='constant'))
      #print("-----=-=-=-pdflow_input",pdflow_input.shape)
      pdflow_input = pdflow_input.view(
        (batch_size*target_length,) + (3,224,224)
        ).to(device)
      #print("-----=-=-=-pdflow_input",pdflow_input.shape)
      encodercnn_output = models['encodercnn'](pdflow_input)
    
  encodercnn_output = encodercnn_output.view(
    (batch_size, target_length) + models['encodercnn'].out_size )
  encoder_output, _ = models['encoder'](encodercnn_output) # (batch, seq_len, c, h, w)
  if split == 'test':
    result['output']['encoder_output'] = encoder_output
  encoder_output = encoder_output.contiguous().view(
    (batch_size*target_length,) + models['encoder'].out_size[1:] )



  # ActionClassifier
  actionclassify_output = models['actionclassifier'](
    encoder_output.view(batch_size*target_length,-1) 
    )
  #print("actionclassify_output=",actionclassify_output.shape,"\n")
  actionclassify_output = actionclassify_output.view(
    (batch_size, target_length) + (models['actionclassifier'].num_classes,) )
  #print("actionclassify_output=",actionclassify_output.shape,"\n")

  if split in ['train', 'validate']:
    if set_grad:
      if 'encodercnn' in target_modules: optimizers['encodercnn'].zero_grad()
      if 'encoder' in target_modules: optimizers['encoder'].zero_grad()
      if 'actionclassifier' in target_modules: optimizers['actionclassifier'].zero_grad() #actionaction needed
    

    action_accuracy = 0
    action_loss = 0
    correct_action = 0
    actionclassify_mean = 0

      
    #actionclassify_output = torch.autograd.Variable(torch.mean(actionclassify_output, 1, True).repeat(1, target_length, 1) ,requires_grad=True)
    
    actionclassify_output = actionclassify_output.mean(1)

    
    #print("===========actionclassify_output=",actionclassify_output.shape) # ([16, 6, 60])
    #print("actionclassify_output=",actionclassify_output)
    #print("actionclassify_output.shape=",actionclassify_output.shape,"\n")
    
    if split in ['validate']:
      ##actionclassify_output = actionclassify_output.mean(0)
      ###print("actionclassify_output.shape=",actionclassify_output.view(1,60).shape,"\n")
      ##action_loss = criterions['actionclassify'](actionclassify_output.view(1,60), sample['action_label'].to(device))
      #print("=============sample[action_label]",sample['action_label'].view((10,)).shape)
      action_loss = criterions['actionclassify'](actionclassify_output, sample['action_label'].view((10,)).to(device))
      #print("===========action_loss=",action_loss.shape)
      action_accuracy = criterions['action_accuracy'](actionclassify_output.to(device))
      #print("===========action_accuracy=",action_accuracy.shape)
      action_accuracy = action_accuracy.mean(0)
      #print("===========action_accuracy_mean(0)=",action_accuracy.shape)
      actionclassify_max_idx = torch.argmax(action_accuracy, 0, keepdim=False) #2, keepdim=False)
      #print("===========actionclassify_max_idx=",actionclassify_max_idx.shape)
      action_IDs = sample['action_label'][0]
    else:
      #print("=============sample[action_label]",sample['action_label'].shape)
      action_loss = criterions['actionclassify'](actionclassify_output, sample['action_label'].view(batch_size).to(device))
      #print("===========action_loss=",action_loss.shape)
      action_accuracy = criterions['action_accuracy'](actionclassify_output.to(device))
      #print("===========action_accuracy=",action_accuracy.shape)
      actionclassify_max_idx = torch.argmax(action_accuracy, 1, keepdim=False) #2, keepdim=False)
      #print("===========actionclassify_max_idx=",actionclassify_max_idx.shape)
      action_IDs = sample['action_label']
    
    #print("actionclassify_output=",actionclassify_output.shape,"\n",actionclassify_output)
    #actionclassify_output.data = actionclassify_output.data.repeat(1, target_length, 1) # ([16, 6, 60])
    #print("actionclassify_output=",actionclassify_output.shape)#,"\n",actionclassify_output)

    
    #print("action_loss=",action_loss.shape,action_loss)

    
    #action_loss.register_hook(lambda grad: print(grad))

    
    #print("---------------------action_accuracy=",action_accuracy.shape)  # ([16, 6, 60]) # batch, targlen, classes [16,60]
    
    #print("sample['action_label']\n",sample['action_label'].shape)
    
    #print("action_accuracy=",action_accuracy.shape,"\n",action_accuracy)
    ##############
    #action_acc_mean_T = action_accuracy.mean(1)
    #print("action_acc_mean_T=",action_acc_mean_T.shape,"\n",action_acc_mean_T) # ([16, 60])
    ##############
    
      
    
    #print("actionclassify_max_idx=",actionclassify_max_idx.shape,actionclassify_max_idx) # ([16])
    
    
    #print("=====action_IDs=",action_IDs.shape, action_IDs) # ([16])
    
    #action_true_preds = [action_IDs[i].repeat(actionclassify_max_idx.shape[-1]) for i in range(actionclassify_max_idx.shape[0])]
    #print("action_true_preds=",len(action_true_preds),action_true_preds) 
    
    
    correct_action =  (torch.sum(  torch.eq(actionclassify_max_idx.float(), action_IDs.to(device))  ) )/float(action_IDs.shape[0])
    #print("correct_action=",correct_action.shape, 100. * correct_action) 
    
    action_accuracy = 100. * correct_action
 
    
    print("=====================================================================  action_loss",action_loss.item()," action_accuracy=",action_accuracy.item())
    
    if set_grad and action_loss != 0:
      action_loss.backward()
      if 'encodercnn' in target_modules: optimizers['encodercnn'].step()
      if 'encoder' in target_modules: optimizers['encoder'].step()
      if 'actionclassifier' in target_modules: optimizers['actionclassifier'].step() 

    result['logs']['actionclassify'] = action_loss.item() if action_loss > 0 else 0
    result['logs']['actionrecognition_accuracy'] = action_accuracy.item() if action_accuracy > 0 else 0

  return result



def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  
  # Input Modality
  parser.add_argument('--modality', dest='modality', 
    default='rgb', help='RGB,Depth,PDFlow')

  # argument for adding comments
  parser.add_argument('--comment', dest='comment', 
    default='', help='just enter additional network details here')

  # What To Do
  parser.add_argument('--train', dest='for_what', 
    action='store_const', const='train', default='train', help='Train')
  parser.add_argument('--trainAction', dest='for_what', 
    action='store_const', const='trainAction', default='trainAction', help='TrainAction')
  parser.add_argument('--test', dest='for_what', 
    action='store_const', const='test', help='Test')
  parser.add_argument('--target-modules', dest='target_modules', 
    default=ALL_MODELS, nargs="*", choices=ALL_MODELS, 
    help='Modules to train or test')

  # Input Directories and Files
  parser.add_argument('--ntu-dir', dest='ntu_dir',
    default='./dataset/NTU_RGB+D_processed/',
    help='Directory that contains json, labels, rgb_h5_dir, depth_h5_dir, and'
         'flow_h5_dir directories')
  parser.add_argument('--checkpoint-files', default='{ }',
    type=json.loads, 
    help='JSON string to indicate checkpoint file for each module. '
         'Beside module names like encoder, and viewclassifier, you can use '
         'special name "else", and "all".'
         'Example: {"encoder": "encodercheck.tar", "else": "allcheck.tar"}')

  # Output Directories
  parser.add_argument('--save-dir', dest='save_dir', default='./VIAR',
    help='Directory to save checkpoints and logs')
  parser.add_argument('--unique-name', dest='unique_name',
    default=None, help='Uniquely names directory within save_dir')
  parser.add_argument('--output-dir', dest='output_dir',
    default='./VIAR/features', help='Output directory for outputs, e.g. extracted features.')

  # Networks
  parser.add_argument('--encoder-cnn-model', dest='encoder_cnn_model',
    default='resnet18', help='Choose between options, for CNN inside Encoder',
    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
  parser.add_argument('--encoder-hid-size', dest='encoder_hid_size',
    type=int, default=64, help='Hidden size of ConvBiLSTM inside Encoder')
  parser.add_argument('--disable-grl', dest='disable_grl',
    default=False, action='store_true',
    help='Disable Gradient Reversal Layer inside ViewClassifier')

  # Training Parameters
  parser.add_argument('--batch-size', dest='batch_size',
    type=int, default=1, help='Input minibatch size')
  parser.add_argument('--learning-rate', dest='learning_rate',
    type=float, default=1e-5, help='Learning rate for training')
  parser.add_argument('--learning-rate-decay', dest='learning_rate_decay',
    type=float, default=50000, help='Learning rate decay for training')
  parser.add_argument('--val-every-iter', dest='val_every_iter',
    type=int, default=None, 
    help='Run validation every val_every_iter iterations. If None, validation '
         'is done after training set iteration.')
  parser.add_argument('--val-size', dest='val_size',
    type=int, default=None, 
    help='Size of minibatches for each run of validation. If None, all samples'
         ' will be used for validation.')
  parser.add_argument('--record-every-iter', dest='record_every_iter',
    type=int, default=1, 
    help='Print and log to tensorboard every record_every_iter iteration')

  # Dataloaders
  parser.add_argument('--visual-transform', dest='visual_transform',
    default='normalize', choices=[None, 'normalize'],
    help='Transform to apply in NTURGBDwithFlow')
  parser.add_argument('--target-length', dest='target_length',
    type=int, default=6,
    help='Length of sequences (frames) used by networks. Will be uniformly '
         'sampled within each video.')
  parser.add_argument('--num-workers', dest='num_workers', default=1,
    type=int, help='Number of workers to load train/test data samples')

  args = parser.parse_args()
  
  params = str(vars(args))

  s = params.find('[') # handle a list
  e = params.find(']', s)
  params = params[:s] + params[s:e+1].replace(', ', ' ') + params[e+1:]
  s = params.find('checkpoint_files\': {') # handle a dict
  e = params.find('}', s)
  params = params[:s] + params[s:e+1].replace(', ', ' ') + params[e+1:]
  params = sorted( params[1:-1].replace("'","").split(', ') )
  print( "\nRunning with following parameters: \n  {}\n".format('\n  '.join(params)) )

  return args

if __name__ == "__main__":
  main()