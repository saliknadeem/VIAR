#!/usr/bin/env python3
"""
References
[1] J. Li, Y. Wong, Q. Zhao, and M. S. Kankanhalli, “Unsupervised Learning of 
    View-invariant Action Representations.," NeurIPS, 2018.
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchvision

import ConvolutionalRNN
import convLSTM
from RevGrad import RevGrad

import math


class CNN(nn.Module):
  def __init__(self, input_shape, model_name='resnet18', input_channel=3):
    super(CNN, self).__init__()
    self.model_name = model_name
    self.out_size = None

    #CNN
    if self.model_name.startswith('resnet'):
      self.front_model = getattr(torchvision.models, self.model_name)(pretrained=False)
      #print("-------input_shape",input_shape)
      if input_channel != 3:
        self.front_model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, 
                                           stride=2, padding=3, bias=False)
                                           #stride=2, padding=3, bias=False) #skl
        self.front_model.conv1.apply(self._init_weights)

      #print('=================stuff===',*list(self.front_model.children())[:-2])
      self.front_model = nn.Sequential(*list(self.front_model.children())[:-2])
      last_conv_outsize = self._get_intermediate_outsize(input_shape, interrupt=1)
      #print("---------last_conv_outsize",last_conv_outsize)
      # Add a 1 × 1 × 64 convolutional layer to reduce the feature size,
      # following [1]
      self.front_model = nn.Sequential(
        self.front_model, 
        nn.Conv2d(last_conv_outsize[0], 64, kernel_size=1, bias=False)
        )
      self.out_size = self._get_intermediate_outsize(input_shape, interrupt=1)
    else:
      raise NotImplementedError('model_name {} not implemented.'.format(self.model_name))

  def forward(self, x, interrupt=0):
    if self.model_name.startswith('resnet'):
      #print("x=====",x.shape)
      x = self.front_model( x )
      #print("x=====",x.shape)
    if interrupt == 1: 
      # _get_intermediate_outsize
      return x

    return x

  def _init_weights(self, m):
    if type(m) == nn.Linear:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      m.bias.data.fill_(0.01)
      #m.bias.data.fill_(0.01) #skl
    else:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  # generate input sample and forward to get output shape
  def _get_intermediate_outsize(self, input_shape, interrupt):
    input = Variable(torch.rand(1, *input_shape)) # 1 for batch_size
    #print("====input",input.shape)
    output_feat = self.forward(input, interrupt=interrupt)
    #print("====output_feat",output_feat.shape)
    n_size = output_feat.size()[1:]
    #print("====n_size",n_size)
    #exit()
    return n_size

class Encoder(nn.Module):
  def __init__(self, input_shape, encoder_block='convbilstm', hidden_size=64):
    super(Encoder, self).__init__()
    self.input_shape = input_shape
    self.encoder_block=encoder_block
    self.hidden_size = hidden_size
    self.out_size = None
    #self.num_layers = 1
    
    #self.hidden_size = hidden_size
    #self.num_layers = num_layers

    #self.fc = nn.Linear(hidden_size * 2, num_classes)

    # Encoder
    if self.encoder_block == 'convbilstm':
      self.convlstm = ConvolutionalRNN.Conv2dLSTM(
        in_channels=self.input_shape[0],  # Corresponds to input size
        out_channels=self.hidden_size,  # Corresponds to hidden size
        kernel_size=7,  # Int or List[int]
        #num_layers=1, stride=1, dropout=0.2, #### skl it was 1
        bidirectional=True,
        #dilation=2,stride=1, dropout=0.2,
        #dropout=0.1,
        batch_first=True
        )
    elif self.encoder_block == 'brnn':
        #print("self.input_shape[0]====",self.input_shape[0])
        self.convlstm = nn.LSTM(
            self.input_shape[0], self.hidden_size, self.num_layers, batch_first=True, bidirectional=True
            )
      # self.convlstm.apply(self._init_weights) # TODO
    else:
      raise NotImplementedError(
        'Given argument encoder_block: {} is not implemented.'.format(self.encoder_block)
        )

    # indicate out_size according to what you are going to do
    if self.encoder_block == 'convbilstm' or self.encoder_block == 'brnn':
      self.out_size = self._get_intermediate_outsize(input_shape, interrupt=1)
      #print("brnn/bilstm self.out_size==",self.out_size)

  def _init_weights(self, m):
    #print("type(m)",type(m))
    if type(m) == nn.Linear:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      m.bias.data.fill_(0.01)
      #m.bias.data.fill_(0.01)
    elif type(m) == ConvolutionalRNN.Conv2dLSTM:
      for weight in m.parameters():
        nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
    elif type(m) == nn.LSTM:
      for weight in m.parameters():
        nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')        
    else:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def forward(self, x, interrupt=0):
    if self.encoder_block == 'convbilstm':
      # Hidden states are initialized automatically when None is given
      hidden = None

      # Go through Encoder
      # Unsqueeze x to set batch_size as 1 and sequence length as the number
      #   of frames. 
      # Input  shape: (batch, seq_len, input_size)
      # Output shape: (batch, seq_len, num_directions * hidden_size)
      # Hidden shape: (batch, num_layers * num_directions, hidden_size)
      output, hidden = self.convlstm(x, hidden)
    elif self.encoder_block == 'brnn':
      h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
      c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

      output, _ = self.convlstm(x, (h0, c0))

    if interrupt == 1: 
        # _get_intermediate_outsize
      return output

    return output, hidden

  # generate input sample and forward to get output shape
  def _get_intermediate_outsize(self, input_shape, interrupt):
    input = Variable(torch.rand(1, 1, *input_shape)) # 1, 1 for batch_size and seq_len
    #print("input to brnn",input.shape)
    output_feat = self.forward(input, interrupt=interrupt)
    n_size = output_feat.size()[1:]
    return n_size




class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out






##########################################################


'''
########################################################
class BiLSTM(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            dropout = dropout,
                            bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*lstm_layer*2, 1)
    
    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.hidden2label(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))
        return y

########################################################

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

########################################################



'''


class CrossViewDecoder(nn.Module):
  def __init__(self, input_shape):
    super(CrossViewDecoder, self).__init__()
    self.input_shape = input_shape
    self.out_size = None

    '''    # Transposed Conv
    self.deconv2d_1 = nn.ConvTranspose2d(
      in_channels=self.input_shape[0], out_channels=80, kernel_size=3, 
      stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)
      #stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1) #skl
    self.deconv2d_1_bn = nn.BatchNorm2d(80) #skl

    self.deconv2d_2 = nn.ConvTranspose2d(
      in_channels=80, out_channels=36, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    self.deconv2d_2_bn = nn.BatchNorm2d(36) #skl 36

    self.deconv2d_3 = nn.ConvTranspose2d(
      in_channels=36, out_channels=17, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    self.deconv2d_3_bn = nn.BatchNorm2d(17) #skl 17

    self.deconv2d_4 = nn.ConvTranspose2d(
      in_channels=17, out_channels=3, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

    # indicate out_size according to what you are going to do
    self.out_size = self._get_intermediate_outsize(input_shape, interrupt=1)

    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)'''


    '''    self.dfc3 = nn.Linear(input_shape, 1024)
    self.bn3 = nn.BatchNorm2d(1024)
    self.dfc2 = nn.Linear(1024, 1024)
    self.bn2 = nn.BatchNorm2d(1024)
    self.dfc1 = nn.Linear(1024,256 * 6 * 6)
    self.bn1 = nn.BatchNorm2d(256*6*6)
    self.upsample1=nn.Upsample(scale_factor=2)'''

    self.upsample1=nn.Upsample(scale_factor=2)
    self.dconv5 = nn.ConvTranspose2d(input_shape[0], 128, 3, 
      stride=2, padding=2, output_padding=1, groups=1, bias=True, dilation=1)
    self.bn5 = nn.BatchNorm2d(128)
    self.dconv4 = nn.ConvTranspose2d(128, 128, 7, padding = 1)
    self.bn4 = nn.BatchNorm2d(128)
    self.dconv3 = nn.ConvTranspose2d(128, 80, 5, padding = 1)
    self.bn3 = nn.BatchNorm2d(80)
    self.dconv2 = nn.ConvTranspose2d(
      in_channels=80, out_channels=30, kernel_size=5, padding=0)
    self.bn2 = nn.BatchNorm2d(30)
    self.dconv1 = nn.ConvTranspose2d(in_channels=30, out_channels=3, kernel_size=7)
    # indicate out_size according to what you are going to do
    self.out_size = self._get_intermediate_outsize(input_shape, interrupt=1)

    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)




  def forward(self, x, e, interrupt=0):
    # e is the output of Encoder to be concatenated with x
    # Shapes:
    #   x: (k, h, w)
    #   e: (2k, h, w)
    #   x concat e: (3k, h, w)
    
    '''    x = torch.cat((x,e), dim=1) # dim 0 is batch dimension
    x = F.relu( self.deconv2d_1_bn( self.deconv2d_1(x) ) )
    x = F.relu( self.deconv2d_2_bn( self.deconv2d_2(x) ) )
    x = F.relu( self.deconv2d_3_bn( self.deconv2d_3(x) ) )
    x = self.deconv2d_4(x) # x.size(0) for batch size'''
    
    #print ('x-base===',x.size())
    x = torch.cat((x,e), dim=1)
    #print ('x-cat===',x.size())
    x = F.relu(self.bn5( self.dconv5(x) ))
    #x=self.upsample1(x)
    #print ('x-c5===',x.size())
    x = F.relu(self.bn4( self.dconv4(x) ))
    #print ('x-c4===',x.size())
    x = F.relu(self.bn3( self.dconv3(x) ))
    #print ('x-c3===',x.size())
    #x=self.upsample1(x)
    #print (x.size())
    x = F.relu(self.bn2( self.dconv2(x) ))
    #print ('x-c2===',x.size())
    #x = F.relu(x)
    #x=self.upsample1(x)
    #print (x.size())
    x = self.dconv1(x)
    #print ('x-c1===',x.size())
 
    
    if interrupt == 1: 
      # _get_intermediate_outsize
      return x

    return x

  # generate input sample and forward to get output shape
  def _get_intermediate_outsize(self, input_shape, interrupt):
    input1_shape = (input_shape[0]//3,) + input_shape[1:]
    input2_shape = (input_shape[0]*2//3,) + input_shape[1:]
    input = Variable(torch.rand(1, *input1_shape)) # 1 for batch_size
    input2 = Variable(torch.rand(1, *input2_shape)) # 1 for batch_size
    output_feat = self.forward(input, input2, interrupt=interrupt)
    n_size = output_feat.size()[1:]
    return n_size

class ReconstructionDecoder(nn.Module):
  def __init__(self, input_shape):
    super(ReconstructionDecoder, self).__init__()
    self.input_shape = input_shape
    self.out_size = None

    # Transposed Conv
    self.deconv2d_1 = nn.ConvTranspose2d(
      in_channels=self.input_shape[0], out_channels=64, kernel_size=3, 
      stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1)
    self.deconv2d_1_bn = nn.BatchNorm2d(64)

    self.deconv2d_2 = nn.ConvTranspose2d(
      in_channels=64, out_channels=32, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    self.deconv2d_2_bn = nn.BatchNorm2d(32)

    self.deconv2d_3 = nn.ConvTranspose2d(
      in_channels=32, out_channels=16, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    self.deconv2d_3_bn = nn.BatchNorm2d(16)

    self.deconv2d_4 = nn.ConvTranspose2d(
      in_channels=16, out_channels=3, kernel_size=5, 
      stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

    # indicate out_size according to what you are going to do
    self.out_size = self._get_intermediate_outsize(input_shape, interrupt=1)

    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x, interrupt=0):
    x = F.relu( self.deconv2d_1_bn( self.deconv2d_1(x) ) )
    x = F.relu( self.deconv2d_2_bn( self.deconv2d_2(x) ) )
    x = F.relu( self.deconv2d_3_bn( self.deconv2d_3(x) ) )
    x = self.deconv2d_4(x) # x.size(0) for batch size
    if interrupt == 1: 
      # _get_intermediate_outsize
      return x

    return x

  # generate input sample and forward to get output shape
  def _get_intermediate_outsize(self, input_shape, interrupt):
    input = Variable(torch.rand(1, *input_shape)) # 1 for batch_size
    output_feat = self.forward(input, interrupt=interrupt)
    n_size = output_feat.size()[1:]
    return n_size

class ViewClassifier(nn.Module):
  def __init__(self, input_size, num_classes, reverse=True):
    super(ViewClassifier, self).__init__()
    self.num_classes = num_classes
    
    # Gradient Reversal Layer with two 
    self.grl = RevGrad(reverse=reverse)
    self.fc1 = nn.Linear(input_size, 1024)
    self.fc2 = nn.Linear(1024, self.num_classes)

    self.fc1.apply(self._init_weights)
    self.fc2.apply(self._init_weights)

  def _init_weights(self, m):
    if type(m) == nn.Linear:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      m.bias.data.fill_(0.01)
    else:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def forward(self, x):
    x = self.grl(x)
    x = F.relu( self.fc1( x ) )
    x = self.fc2(x)
    return x


class ActionClassifier(nn.Module):
  def __init__(self, input_size, num_classes):
    super(ActionClassifier, self).__init__()
    self.num_classes = num_classes
    
    self.fc1 = nn.Linear(input_size, self.num_classes)
    #self.fc2 = nn.Linear(1024, self.num_classes)

    self.fc1.apply(self._init_weights)
    #self.fc2.apply(self._init_weights)

  def _init_weights(self, m):
    if type(m) == nn.Linear:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      m.bias.data.fill_(0.01)
    else:
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def forward(self, x):
    #x = F.relu( self.fc1( x ) )
    x = self.fc1(x)
    return x





class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, models):
        super(MultiTaskLossWrapper, self).__init__()
        self.models = models
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, sample, criterions,encoder_output, device='cuda'):
        #task_num = 4
        #log_vars = torch.zeros((task_num))
        #loss = 0
        DEPTH_INPUT_SHAPE = (1,224,224)
        batch_size = len(sample['videoname'])
        target_length = len(sample['rgbs'][0])
        
        encoder_output = encoder_output.contiguous().view(
        (batch_size*target_length,) + self.models['encoder'].out_size[1:] )

        # CrossViewDecoder
        otherview_depth_input = sample['otherview_depths'].view(
          (batch_size*target_length,) + DEPTH_INPUT_SHAPE ).to(device)
        otherview2_depth_input = sample['otherview2_depths'].view(
          (batch_size*target_length,) + DEPTH_INPUT_SHAPE ).to(device) #skl

        crossviewcnn_output = self.models['crossviewdecodercnn'](otherview_depth_input)
        crossviewcnn_output2 = self.models['crossviewdecodercnn'](otherview2_depth_input) #skl
        crossview_output = self.models['crossviewdecoder'](crossviewcnn_output, encoder_output)
        crossview_output2 = self.models['crossviewdecoder'](crossviewcnn_output2, encoder_output) #skl
        crossview_output = crossview_output.view(
          (batch_size, target_length) + self.models['crossviewdecoder'].out_size )
        crossview_output2 = crossview_output2.view(
          (batch_size, target_length) + self.models['crossviewdecoder'].out_size ) #skl



        # ReconstructionDecoder
        reconstruct_output = self.models['reconstructiondecoder'](encoder_output)
        reconstruct_output = reconstruct_output.view(
          (batch_size, target_length) + self.models['reconstructiondecoder'].out_size )


        # ViewClassifier
        viewclassify_output = self.models['viewclassifier'](
          encoder_output.view(batch_size*target_length,-1) )
        viewclassify_output = viewclassify_output.view(
          (batch_size, target_length) + (self.models['viewclassifier'].num_classes,) )
        
        viewclassify_loss = criterions['viewclassify'](viewclassify_output, sample['view_id'].long().to(device))
        
        
        '''loss = torch.sum(torch.exp(-self.log_vars[0]).to(device) * (sample['otherview_flows'].to(device) - crossview_output) ** 2. + self.log_vars[0], -1).to(device)
        loss += torch.sum(torch.exp(-self.log_vars[1]).to(device) * (sample['otherview2_flows'].to(device) - crossview_output2) ** 2. + self.log_vars[1], -1).to(device)
        loss += torch.sum(torch.exp(-self.log_vars[2]).to(device) * (sample['flows'].to(device) - reconstruct_output) ** 2. + self.log_vars[2], -1).to(device)
        loss += torch.sum(torch.exp(-self.log_vars[3]).to(device) * -math.log(criterions['view_accuracy'](viewclassify_loss) )  + self.log_vars[3], -1).to(device)  
        loss = torch.mean(loss).to(device)
'''
        
        #loss = torch.sum(torch.exp(-self.log_vars[0]).to(device) * 
        #                criterions['crossview'](crossview_output, sample['otherview_flows'].to(device)) + self.log_vars[0], -1).to(device)
        #loss += torch.sum(torch.exp(-self.log_vars[1]).to(device) * 
        #                 criterions['crossview'](crossview_output2, sample['otherview2_flows'].to(device)) + self.log_vars[1], -1).to(device)
        loss = torch.sum(torch.exp(-self.log_vars[2]).to(device) * 
                          criterions['reconstruct'](reconstruct_output, sample['flows'].to(device)) + self.log_vars[2], -1).to(device)
        loss += torch.sum(torch.exp(-self.log_vars[3]).to(device) * 
                          -math.log(criterions['view_accuracy'](viewclassify_loss) )  + self.log_vars[3], -1).to(device)  
        
        loss = torch.mean(loss).to(device)

        return loss, self.log_vars.data.tolist()

    
    