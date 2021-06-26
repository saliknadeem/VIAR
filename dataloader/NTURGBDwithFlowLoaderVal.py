#!/usr/bin/env python3
"""
References
[1] J. Li, Y. Wong, Q. Zhao, and M. S. Kankanhalli, “Unsupervised Learning of 
    View-invariant Action Representations.," NeurIPS, 2018.
"""
import os
import json
import h5py
import random
import operator
import collections
import numpy as np
from PIL import Image


import cv2
import skimage.measure
from skimage.transform import resize


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F

class NTURGBDwithFlow(Dataset):
  """
    NTU RGB+D dataset with preextracted 3D flow for Viewpoint-Invariant Scene
  Identification.
  """

  def __init__(
    self, 
    json_file, 
    label_file, 
    rgb_h5_dir, 
    depth_h5_dir, 
    flow_h5_dir, 
    target_length, 
    subset, 
    visual_transform='normalize'
    ):
    """
    Args:
      json_file (string): Path to the json file with videonames and number of 
        frames for each videoname.
      label_file (string): Path to the text file with action label texts where
        line number corresponds to the action index.
      rgb_h5_dir (string): Directory with h5 files containing 8-bit rgb PNG 
        files as bytes.
      depth_h5_dir (string): Directory with h5 files containing 16-bit 
        monochrome depth images as bytes.
      flow_h5_dir (string): Directory with h5 files containing extracted 3d 
        flows in float.
      target_length (int): The number of frames to be returned for each video.
      subset (string): 'all', 'train', or 'test'.
      visual_transform (string): None or 'normalize'. If 'normalize', rgb images
        are normalized with ImageNet statistics.
    """
    with open(json_file, 'r', encoding='utf-8') as fp:
      self.meta = json.load(fp, object_pairs_hook=collections.OrderedDict)
    with open(label_file, 'r') as f:
      self.labels = [line.strip() for line in f.readlines()]
    self.num_classes = 60 

    self.rng = random.SystemRandom()
    
    self.rgb_h5_dir = rgb_h5_dir
    self.depth_h5_dir = depth_h5_dir
    self.flow_h5_dir = flow_h5_dir
    self.target_length = target_length
    self.subset = subset
    self.visual_transform = visual_transform

    self.side_size = 224 # width and height to be returned
    self.patch_size = 8 # for downsampled flow images

    self.videonames = []
    self.videonames_sequences = []
    """
      Videoname looks like  in the format of SsssCcccPpppRrrrAaaa 
    (e.g., S001C002P003R002A013), in which sss is the setup number, ccc is 
    the camera ID, ppp is the performer (subject) ID, rrr is the replication 
    number (1 or 2), and aaa is the action class label.
    """
    
    """
      Videoname_sequences looks like  in the format of SsssCcccPpppRrrrAaaaFfff 
    (e.g., S001C002P003R002A013F001), in which sss is the setup number, ccc is 
    the camera ID, ppp is the performer (subject) ID, rrr is the replication 
    number (1 or 2), and aaa is the action class label. fff is the tensor unit 
    based on target-length, it divides the frames in consecutive sequences from 
    one action video.
    """
    
    
    
    """ Old code
    for videoname in self.meta['videonames']:
      if self.subset == 'all':
        self.videonames.append(videoname)
      elif self.subset == 'train':
        if int(videoname[9:12]) in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]:
          # Cross-Subject Evalutation subjects (perferomers id), 40,320 samples
          self.videonames.append(videoname)
      else: # test set
        if int(videoname[9:12]) not in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]:
          # 16,560 samples
          self.videonames.append(videoname)
    self.videonames = sorted(self.videonames)
    """
    
    
    """
    Addind new code here
    """
    
    
    ActionClasses = np.zeros(60)
    Subjects = np.zeros(40)
    ViewIDs = np.zeros(5)
    CamIDs = np.zeros(3)
    
    for videoname in self.meta['videonames']:
      if self.subset == 'all':
        self.videonames.append(videoname)
        
        length = self.meta['framelength'][videoname]-5
        #seq_list = list (range(1, (length-(length%self.target_length))//self.target_length+1))
        seq_size = (length-(length%self.target_length))//self.target_length
        for f in range(1,seq_size+1):
            self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))
      
      elif self.subset == 'train':
        if int(videoname[9:12]) in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]:
            
            
            if False: # New validation set with 10 equal sequences for all
                self.videonames.append(videoname)
                seq_size = 10
                for f in range(1,seq_size+1):
                    self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))
                
            else:
                # Cross-Subject Evalutation subjects (perferomers id), 40,320 samples
                self.videonames.append(videoname)

                length = self.meta['framelength'][videoname]-5

                #seq_list = list (range(1, (length-(length%self.target_length))//self.target_length+1))
                seq_size = (length-(length%self.target_length))//self.target_length
                for f in range(1,seq_size+1):
                    self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))

                """######## Class balance check
                camera_id = int(videoname[5:8])
                replication_id = int(videoname[13:16])

                #   Based on camera id and replication id, calculate view id for view 
                # classification. 
                # 1: front view, 2: left view (90 deg), 3: right view (90 deg), 
                # 4: left view (45 deg), 5: right view (45 deg)
                if replication_id == 1:
                  if camera_id == 2:
                    view_id = 1
                  elif camera_id == 1:
                    view_id = 5
                  elif camera_id == 3:
                    view_id = 3

                elif replication_id == 2:
                  if camera_id == 2:
                    view_id = 2
                  elif camera_id == 1:
                    view_id = 4
                  elif camera_id == 3:
                    view_id = 1

                action_id = int(videoname[17:20])
                setup_id = int(videoname[1:4])
                subject_id = int(videoname[9:12])

                ActionClasses[action_id-1] += seq_size
                Subjects[subject_id-1] += seq_size
                ViewIDs[view_id-1] += seq_size
                CamIDs[camera_id-1] += seq_size
                #######
                """
    
      else: # test set
        if int(videoname[9:12]) not in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]:
            if False: # New validation set with 10 equal sequences for all
                self.videonames.append(videoname)
                seq_size = 10
                for f in range(1,seq_size+1):
                    self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))

            else:
                # 16,560 samples
                self.videonames.append(videoname)

                length = self.meta['framelength'][videoname]-5
                #seq_list = list (range(1, (length-(length%self.target_length))//self.target_length+1))
                seq_size = (length-(length%self.target_length))//self.target_length
                for f in range(1,seq_size+1):
                    self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))


                ''' Class balance check ##############
                camera_id = int(videoname[5:8])
                replication_id = int(videoname[13:16])

                #   Based on camera id and replication id, calculate view id for view 
                # classification. 
                # 1: front view, 2: left view (90 deg), 3: right view (90 deg), 
                # 4: left view (45 deg), 5: right view (45 deg)
                if replication_id == 1:
                  if camera_id == 2:
                    view_id = 1
                  elif camera_id == 1:
                    view_id = 5
                  elif camera_id == 3:
                    view_id = 3

                elif replication_id == 2:
                  if camera_id == 2:
                    view_id = 2
                  elif camera_id == 1:
                    view_id = 4
                  elif camera_id == 3:
                    view_id = 1

                action_id = int(videoname[17:20])
                setup_id = int(videoname[1:4])
                subject_id = int(videoname[9:12])

                ActionClasses[action_id-1] += seq_size
                Subjects[subject_id-1] += seq_size
                ViewIDs[view_id-1] += seq_size
                CamIDs[camera_id-1] += seq_size
                '''
    
    print("=======actual videonames===========",len(self.videonames))
    print("========total data==========",len(self.videonames_sequences))
    self.videonames = sorted(self.videonames)
    self.videonames_sequences = sorted(self.videonames_sequences)
    
    '''
    print('-------------------------------------',self.subset)            
    print("ActionClasses=\n",ActionClasses*100./np.sum(ActionClasses))
    print("Subjects=\n",Subjects*100./np.sum(Subjects))
    print("ViewIDs=",ViewIDs*100./np.sum(ViewIDs))
    print("CamIDs=",CamIDs*100./np.sum(CamIDs))
    print('----------------------------------------------------------------')
    '''
    
    #print("total videos------------------------",len(self.videonames))
    #print("total new stuff------------------------",len(self.videonames_sequences))
    #exit()
    
    if self.visual_transform is None:
      self.rgb_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              #transforms.CenterCrop(self.side_size),
                              transforms.Resize((self.side_size,self.side_size),interpolation=Image.NEAREST),
                              transforms.ToTensor()
                              ])
      '''self.flow_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.CenterCrop(self.side_size),
                              transforms.ToTensor()
                              ])'''
    elif self.visual_transform == 'normalize':
      self.rgb_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              #transforms.CenterCrop(self.side_size),
                              transforms.Resize((self.side_size,self.side_size),interpolation=Image.NEAREST),
                              transforms.ToTensor(),
                              #transforms.Normalize((0.5,0.5,0.5),
                              #                     (0.5,0.5,0.5))
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                              ])
      '''self.flow_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.CenterCrop(self.side_size),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[3.1587e-05, -1.7715e-07,  6.3462e-06],
                                                   std=[0.0019, 0.0004, 0.0003])
                              ])'''

    ###print('NTURGBDwithFlow is initialized. '
    ###      '(Subset: {}, Length: {})'.format(self.subset, len(self.videonames_sequences)) )

    print('NTURGBDwithFlow is initialized. '
          '(Subset: {}, Length: {})'.format(self.subset, len(self.videonames_sequences)) )
    
  def __len__(self):
    return len(self.videonames_sequences)
    #return len(self.videonames)

  def __getitem__(self, idx):
    videoname_sequence = self.videonames_sequences[idx]
    videoname = videoname_sequence[0:20]
 

    frame_segment = int(videoname_sequence[21:])
    length = self.meta['framelength'][videoname]
    frame_indices = list(range( (frame_segment*self.target_length)-self.target_length, (frame_segment*self.target_length)))


    
    rgb_h5_path = os.path.join(self.rgb_h5_dir, videoname + '_pngs.h5')
    rgb_h5 = h5py.File(rgb_h5_path, 'r', libver='latest', swmr=True)
    rgbs = []
    ####print("--------------frame_indices------",frame_indices)
    ####print("------------------------------------------",len(rgb_h5['pngs'][:]))
    
    rgb = []
    ####for byte in rgb_h5["pngs"][frame_indices]:
    for byte in rgb_h5["pngs"][frame_indices]:
      rgb = cv2.imdecode(byte, flags=cv2.IMREAD_COLOR)
      rgb = self.rgb_transform(rgb)
      rgbs.append(rgb)
    rgbs = torch.stack(rgbs)

 
    
    #############depth_h5_path = os.path.join(self.depth_h5_dir, videoname + '_pngs.h5')
    depth_h5_path = os.path.join(self.depth_h5_dir, videoname + '_maskeddepth_pngs.h5')
    depth_h5 = h5py.File(depth_h5_path, 'r', libver='latest', swmr=True)
    depths = []
    
    #print("-------------RGB------------", rgb.size())
    #print("-------------RGBs------------", rgbs.size())
    #print("-------------RGBs 0------------", len(rgbs[:]))
    #print("-----------frames--------------", frame_indices)

    for byte in depth_h5['pngs'][frame_indices]:
      #depth = cv2.imdecode(byte, flags=cv2.IMREAD_UNCHANGED)
      #print("------",np.shape(byte))
      #depth = np.array(bytes, dtype=np.uint16)
      #depth = cv2.imread(byte)#, cv2.IMREAD_ANYDEPTH)
      depth = np.array(byte,dtype=np.uint16)
      ##
      depth = cropND(depth, (self.side_size, self.side_size))
      depth = np.expand_dims(depth, axis=0)
      depth = depth.astype(np.float32) / 65535
      depth = torch.FloatTensor(depth)
      depths.append(depth)
    depths = torch.stack(depths)
    
    
    #print("-------------depths------------", depths.size())
    
    '''flow_h5_path = os.path.join(self.flow_h5_dir, videoname + '_3dflow.h5')
    flow_h5 = h5py.File(flow_h5_path, 'r', libver='latest', swmr=True)
    #####flows = []
    flowsActual = []
    for f in flow_h5['flow'][frame_indices]:
      #flow = cropND(f, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
      #flow = skimage.measure.block_reduce(f, (8,8), np.mean)
      flowActual = torch.from_numpy(f)
      #flowActual = cropND(flowActual, (self.side_size, self.side_size, 3))
      flowActual = resize(flowActual, (224, 224))
      flowActual = np.transpose(flowActual, (2,0,1)) 
      flowActual = flowActual * 50
      #flowActual = self.flow_transform(flowActual)
      #flow = torch.from_numpy(f)
     
      
      ##### Resize operation using interpolate on tensors
      #flow = np.transpose(flow, (2, 0, 1))
      #flow = F.interpolate(flow, self.side_size // self.patch_size)
      #flow = np.transpose(flow, (2, 0, 1))
      #flow = F.interpolate(flow, self.side_size // self.patch_size)
      #flow = np.transpose(flow, (1,0,2))
      ######################
      #####flow = np.transpose(flow, (2,0,1))
      
      #flowActual[flowActual != 0] = flowActual[flowActual != 0] + 3.1465
      ##flowActual = flowActual + 3.1465
      #####flow = flow * 1000 # multiply 50 to "keep proper scale" according to [1]
      flowsActual.append(torch.FloatTensor(flowActual))
      #####flows.append(torch.FloatTensor(flow))
    #####flows = torch.stack(flows)
    flowsActual = torch.stack(flowsActual)
    
    #print("-------------flows------------", flows.size())
    
    
    #############################################print("=================actual={}/{}=={}===".format(frame_indices,length,videoname),idx )
    
    flow_sm_h5_path = os.path.join(self.flow_sm_h5_dir, videoname + '_3dflow.h5')
    flow_sm_h5 = h5py.File(flow_sm_h5_path, 'r', libver='latest', swmr=True)
    flows = []
    for f in flow_sm_h5['flow'][frame_indices]:
      flow = torch.from_numpy(f)
      flow = cropND(flow, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
      flow = np.transpose(flow, (2,0,1))
      flow = flow #* 50 # multiply 50 to "keep proper scale" according to [1]
      #flow[flow != 0] = flow[flow != 0] + 1.1980
      flows.append(torch.FloatTensor(flow))
    flows = torch.stack(flows)''' 

    
    flow_h5_path = os.path.join(self.flow_h5_dir, videoname + '_3dflow.h5')
    flow_h5 = h5py.File(flow_h5_path, 'r', libver='latest', swmr=True)
    flows = []
    flowsActual = []
    for f in flow_h5['flow'][frame_indices]:
      #flow = cropND(f, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
      #flow = skimage.measure.block_reduce(f, (8,8), np.mean)
      flowActual = torch.from_numpy(f)
      flow = cv2.resize(f, (self.side_size // self.patch_size, self.side_size // self.patch_size), interpolation = cv2.INTER_AREA)
      flow = torch.from_numpy(flow)
      #flow = resize(f, (self.side_size // self.patch_size, self.side_size // self.patch_size))
      #flow = cropND(flow, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
      
      ##### Resize operation using interpolate on tensors
      '''flow = np.transpose(flow, (0, 2, 1))
      flow = F.interpolate(flow, self.side_size // self.patch_size)
      flow = np.transpose(flow, (0, 2, 1))
      flow = F.interpolate(flow, self.side_size // self.patch_size)'''
      ######################
      flow = np.transpose(flow, (2,0,1))
      flowActual = np.transpose(flowActual, (2,0,1)) 
      #flow[flow<0.0001] = 0
      #flowActual[flowActual<0.0001] = 0
      flowActual = flowActual * 50 # multiply 50 to "keep proper scale" according to [1]
      flow = flow * 50 # multiply 50 to "keep proper scale" according to [1]
      flowsActual.append(torch.FloatTensor(flowActual))
      flows.append(torch.FloatTensor(flow))
    flows = torch.stack(flows)
    flowsActual = torch.stack(flowsActual)
    
    
    


    camera_id = int(videoname[5:8])
    replication_id = int(videoname[13:16])

    #   Based on camera id and replication id, calculate view id for view 
    # classification. 
    # 1: front view, 2: left view (90 deg), 3: right view (90 deg), 
    # 4: left view (45 deg), 5: right view (45 deg)
    if replication_id == 1:
      if camera_id == 2:
        view_id = 1
      elif camera_id == 1:
        view_id = 5
      elif camera_id == 3:
        view_id = 3
      else:
        raise ValueError('Unexpected camera id {} from {}'.format(
                         camera_id, videoname ) )
    elif replication_id == 2:
      if camera_id == 2:
        view_id = 2
      elif camera_id == 1:
        view_id = 4
      elif camera_id == 3:
        view_id = 1
      else:
        raise ValueError('Unexpected camera id {} from {}'.format(
                         camera_id, videoname ) )
    else:
      raise ValueError('Unexpected replication id {} from {}'.format(
                       replication_id, videoname ) )

    # view_id to categorical one-hot vector
    view_id = np.eye(5, dtype='bool')[view_id-1].astype(int) # 5 for 5 views, view_id-1 for number to index
    
    action_id = int(videoname[17:20])
    setup_id = int(videoname[1:4])
    subject_id = int(videoname[9:12])

    other_cameras = [1,2,3]
    other_cameras.remove(camera_id)
    
    otherview_camera_id = other_cameras[0]
    otherview_camera_id2 = other_cameras[1]
    # Get depths images from different camera
    otherview_videoname = 'S{:03d}C{:03d}P{:03d}R{:03d}A{:03d}'.format(
      setup_id, otherview_camera_id, subject_id, replication_id, action_id)
    otherview_videoname2 = 'S{:03d}C{:03d}P{:03d}R{:03d}A{:03d}'.format(
      setup_id, otherview_camera_id2, subject_id, replication_id, action_id)
    
    otherview_videonames = []
    
    
    otherview_videonames.append(otherview_videoname)
    otherview_videonames.append(otherview_videoname2)
    
    
    otherview_length = self.meta['framelength'][otherview_videoname]
    #otherview_length2 = self.meta['framelength'][otherview_videoname2]
    
    
    #print("-------videoname_sequence=", videoname_sequence,"===other===",otherview_videoname,otherview_length,otherview_videoname2,otherview_length2,"length",length,"frame_indices",frame_indices)
    
    
    
    #print("-------otherview_length--------",otherview_length)
    
    #if np.abs(length - otherview_length) >= 5 :
    #    print("length=",length,"- otherview=",otherview_length,"- video=",videoname, "- other-video=",otherview_videoname)
    #    print("frame_indices=", frame_indices)
    
    '''otherview_frame_indices = frame_indices
    
    otherview_depths = []
    otherview_flows = []
    for otherview_videoname_index in otherview_videonames:

        otherview_depth_h5_path = os.path.join(
          self.depth_h5_dir, otherview_videoname_index + '_maskeddepth_pngs.h5')
        otherview_depth_h5 = h5py.File(otherview_depth_h5_path, 'r', libver='latest', swmr=True)
 
        for byte in otherview_depth_h5['pngs'][otherview_frame_indices]:
          depth = cv2.imdecode(byte, flags=cv2.IMREAD_UNCHANGED)
          depth = cropND(depth, (self.side_size, self.side_size))
          depth = np.expand_dims(depth, axis=0)
          depth = depth.astype(np.float32) / 65535
          depth = torch.FloatTensor(depth)
          otherview_depths.append(depth)
        

        #print("-------------otherview_depths------------", otherview_depths.size())   


        otherview_flow_h5_path = os.path.join(
          self.flow_sm_h5_dir, otherview_videoname_index + '_3dflow.h5')
        otherview_flow_h5 = h5py.File(otherview_flow_h5_path, 'r', libver='latest', swmr=True)
        for f in otherview_flow_h5['flow'][otherview_frame_indices]:
          #flow = cropND(f, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
          flow = resize(f, (self.side_size // self.patch_size, self.side_size // self.patch_size))
          flow = np.transpose(flow, (2,0,1))
          flow = flow #* 50 # multiply 50 to "keep proper scale" according to [1]
          #flow[flow != 0] = flow[flow != 0] + 1.1193
          ##flow = flow + 1.1193
          otherview_flows.append(torch.FloatTensor(flow))
        
    
    otherview_depths = torch.stack(otherview_depths)
    otherview_flows = torch.stack(otherview_flows)'''


    
    
    
    otherview_frame_indices = frame_indices
    
    otherview_depths = []
    otherview_flows = []
    for otherview_videoname_index in otherview_videonames:

        otherview_depth_h5_path = os.path.join(
          self.depth_h5_dir, otherview_videoname_index + '_maskeddepth_pngs.h5')
        otherview_depth_h5 = h5py.File(otherview_depth_h5_path, 'r', libver='latest', swmr=True)
 
        for byte in otherview_depth_h5['pngs'][otherview_frame_indices]:
          #depth = cv2.imdecode(byte, flags=cv2.IMREAD_UNCHANGED)
          #depth = cv2.imread(byte)#, cv2.IMREAD_ANYDEPTH)
          depth = np.array(byte,dtype=np.uint16)
          ##
          depth = cropND(depth, (self.side_size, self.side_size))
          depth = np.expand_dims(depth, axis=0)
          depth = depth.astype(np.float32) / 65535
          depth = torch.FloatTensor(depth)
          otherview_depths.append(depth)
        

        #print("-------------otherview_depths------------", otherview_depths.size())   


        otherview_flow_h5_path = os.path.join(
          self.flow_h5_dir, otherview_videoname_index + '_3dflow.h5')
        otherview_flow_h5 = h5py.File(otherview_flow_h5_path, 'r', libver='latest', swmr=True)
        for f in otherview_flow_h5['flow'][otherview_frame_indices]:
          #print("in:",np.shape(flow))
          #flow = cropND(f, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
          flow = cv2.resize(np.asarray(f), (self.side_size // self.patch_size, self.side_size // self.patch_size), interpolation = cv2.INTER_AREA)
          #flow = resize(f, (self.side_size // self.patch_size, self.side_size // self.patch_size))
          flow = np.transpose(flow, (2,0,1))
          #print(np.shape(flow))
          #print(type(flow))
          #flow[flow<0.0001] = 0
          flow = flow * 50 # multiply 50 to "keep proper scale" according to [1]
          otherview_flows.append(torch.FloatTensor(flow))
        
    
    otherview_depths = torch.stack(otherview_depths)
    otherview_flows = torch.stack(otherview_flows)
    
    
    
    ##print("-------------otherview_depths------------", otherview_depths.size())   
    #print("-------------otherview_flows------------", otherview_flows.size())
    ##print("-------------otherview_depths------------", otherview_depths[0:6,:,:,:].size())   
    ##print("-------------otherview_flows------------", otherview_flows[6:,:,:,:].size())
    ##exit()
    
    action_label = videoname[17:].strip("0")
    
    # action_label to categorical one-hot vector
    #action_label = np.eye(60, dtype='bool')[int(action_label)-1].astype(int) # 60 for 60 action classes, action_label-1 for number to index
    action_label = int(action_label)
    
    #print("action_label=",action_label.shape,"\n")
    
    #action_label = action_label.reshape([1,-1]) # ([16, 1, 60])
    #print("action_label=",action_label.shape,"\n")
    #action_label = action_label.repeat(self.target_length, 1) # ([16, 6, 60])
    #print("action_label=",action_label.shape,"\n")
    
    
    
    #print("-------view_id--------",view_id)
    #print("video=",videoname,"\n-------action_label--------",action_label)
    #print("-------torch.Tensor(view_id)--------",torch.Tensor(view_id))
 



    '''    print("-------------videoname------------", videoname)
    print("-------------RGB------------", rgb.size())
    print("-------------RGBs------------", rgbs.size())
    print("-------------depths------------", depths.size())
    print("-------------flows------------", flows.size())
    print("-------------flowsActual------------", flowsActual.size())
    print("-------------otherview_depths------------", otherview_depths[0:6,:,:,:].size())
    print("-------------otherview_flows------------", otherview_depths[0:6,:,:,:].size())
    print("-------------otherview_depths2------------", otherview_depths[6:,:,:,:].size())
    print("-------------otherview_flows2------------", otherview_depths[6:,:,:,:].size())
    print("-------------action_label------------", action_label)'''
    
    sample = {'action_id': action_id,
              'camera_id': camera_id,
              'setup_id': setup_id,
              'subject_id': subject_id,
              'replication_id': replication_id,
              'view_id': torch.Tensor(view_id),
              'rgbs': rgbs,
              'depths': depths,
              'otherview_depths': otherview_depths[0:self.target_length,:,:,:],
              'otherview_flows': otherview_flows[0:self.target_length,:,:,:],
              'otherview2_depths': otherview_depths[self.target_length:,:,:,:],
              'otherview2_flows': otherview_flows[self.target_length:,:,:,:], 
              'action_label': action_label,
              'flows': flows,
              'flowsActual':flowsActual,
              'videoname': videoname,
              'videoname_sequence':videoname_sequence
              }
    

    return sample

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]



def NTURGBDwithFlowLoaderCS(
    json_file, 
    label_file, 
    rgb_h5_dir, 
    depth_h5_dir, 
    flow_h5_dir,
    target_length, 
    subset, 
    visual_transform='normalize', 
    batch_size=1, 
    shuffle=True, 
    num_workers=1, 
    pin_memory=False
    ):

  # load dataset
  dataset = NTURGBDwithFlow(
    json_file=json_file, 
    label_file=label_file, 
    rgb_h5_dir=rgb_h5_dir, 
    depth_h5_dir=depth_h5_dir, 
    flow_h5_dir=flow_h5_dir, 
    target_length=target_length, 
    subset=subset, 
    visual_transform=visual_transform
    )

  # data loader for custom dataset
  data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=pin_memory
    )

  return data_loader, dataset



def NTURGBDwithFlowLoaderCSValidation(
    json_file, 
    label_file, 
    rgb_h5_dir, 
    depth_h5_dir, 
    flow_h5_dir,
    target_length, 
    subset, 
    visual_transform='normalize', 
    batch_size=1, 
    shuffle=False, 
    num_workers=1, 
    pin_memory=False
    ):

  # load dataset
  dataset = NTURGBDwithFlowValidation(
    json_file=json_file, 
    label_file=label_file, 
    rgb_h5_dir=rgb_h5_dir, 
    depth_h5_dir=depth_h5_dir, 
    flow_h5_dir=flow_h5_dir, 
    target_length=target_length, 
    subset=subset, 
    visual_transform=visual_transform
    )

  # data loader for custom dataset
  data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=pin_memory
    )

  return data_loader, dataset





class NTURGBDwithFlowValidation(Dataset):
  """
    NTU RGB+D dataset with preextracted 3D flow for Viewpoint-Invariant Scene
  Identification.
  """

  def __init__(
    self, 
    json_file, 
    label_file, 
    rgb_h5_dir, 
    depth_h5_dir, 
    flow_h5_dir,
    target_length, 
    subset, 
    visual_transform='normalize'
    ):

    with open(json_file, 'r', encoding='utf-8') as fp:
      self.meta = json.load(fp, object_pairs_hook=collections.OrderedDict)
    with open(label_file, 'r') as f:
      self.labels = [line.strip() for line in f.readlines()]
    self.num_classes = 60 

    self.rng = random.SystemRandom()
    
    self.rgb_h5_dir = rgb_h5_dir
    self.depth_h5_dir = depth_h5_dir
    self.flow_h5_dir = flow_h5_dir
    self.target_length = target_length
    self.subset = subset
    self.visual_transform = visual_transform

    self.side_size = 224 # width and height to be returned
    self.patch_size = 8 # for downsampled flow images

    self.videonames = []
    self.videonames_sequences = []
    
    ActionClasses = np.zeros(60)
    Subjects = np.zeros(40)
    ViewIDs = np.zeros(5)
    CamIDs = np.zeros(3)
    
    for videoname in self.meta['videonames']:
      if self.subset == 'all':
        self.videonames.append(videoname)
        
        length = self.meta['framelength'][videoname]-5
        #seq_list = list (range(1, (length-(length%self.target_length))//self.target_length+1))
        seq_size = (length-(length%self.target_length))//self.target_length
        for f in range(1,seq_size+1):
            self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))

        
      elif self.subset == 'train':
        if int(videoname[9:12]) in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]:
            
            
            if False: # New validation set with 10 equal sequences for all
                self.videonames.append(videoname)
                seq_size = 10
                for f in range(1,seq_size+1):
                    self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))
                
            else:
                # Cross-Subject Evalutation subjects (perferomers id), 40,320 samples
                self.videonames.append(videoname)

                length = self.meta['framelength'][videoname]-5

                #seq_list = list (range(1, (length-(length%self.target_length))//self.target_length+1))
                seq_size = (length-(length%self.target_length))//self.target_length
                for f in range(1,seq_size+1):
                    self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))

    
      else: # test set
        if int(videoname[9:12]) not in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]:
            
            if True: # New validation set with 10 equal sequences for all
                self.videonames.append(videoname)
                seq_size = 10
                for f in range(1,seq_size+1):
                    self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))
                
            else:
                # 16,560 samples
                self.videonames.append(videoname)

                length = self.meta['framelength'][videoname]-5
                #seq_list = list (range(1, (length-(length%self.target_length))//self.target_length+1))
                seq_size = (length-(length%self.target_length))//self.target_length
                for f in range(1,seq_size+1):
                    self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))


    print("@@@@@@@=======actual videonames===========",len(self.videonames))
    print("@@@@@@@========total data==========",len(self.videonames_sequences))
    self.videonames = sorted(self.videonames)
    self.videonames_sequences = sorted(self.videonames_sequences)
    

    
    if self.visual_transform is None:
      self.rgb_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              #transforms.CenterCrop(self.side_size),
                              transforms.Resize((self.side_size,self.side_size),interpolation=Image.NEAREST),
                              transforms.ToTensor()
                              ])
      '''self.flow_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.CenterCrop(self.side_size),
                              transforms.ToTensor()
                              ])'''
    elif self.visual_transform == 'normalize':
      self.rgb_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              #transforms.CenterCrop(self.side_size),
                              transforms.Resize((self.side_size,self.side_size),interpolation=Image.NEAREST),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                              ])
      '''self.flow_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.CenterCrop(self.side_size),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[3.1587e-05, -1.7715e-07,  6.3462e-06],
                                                   std=[0.0019, 0.0004, 0.0003])
                              ])'''

    print('NTURGBDwithFlow is initialized. '
          '(Subset: {}, Length: {})'.format(self.subset, len(self.videonames_sequences)) )
    
  def __len__(self):
    #return len(self.videonames_sequences)
    return len(self.videonames)

  def __getitem__(self, idx):

    videoname_sequence = self.videonames_sequences[idx]
    videoname = videoname_sequence[0:20]


    if int(videoname[9:12]) in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]: #Train set
        #frame_segment = int(videoname_sequence[21:])
        length = self.meta['framelength'][videoname]
        #frame_indices = list(range( (frame_segment*self.target_length)-self.target_length, (frame_segment*self.target_length)))
    elif int(videoname[9:12]) not in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]: #Validation set
        #frame_segment = int(videoname_sequence[21:])
        length = self.meta['framelength'][videoname]-5
        total_sequences = length // self.target_length
        total_sequences_list = []
        for i in range(0,length-1):
            if i % self.target_length == 0:
                total_sequences_list.append(list(range(i+1,i+1+self.target_length)))
        frame_indices_temp = np.linspace(1, total_sequences, endpoint=False, 
                                    num=10, dtype=int)
        total_sequences_list = np.array(total_sequences_list)
        frames = total_sequences_list[frame_indices_temp]


    
    rgb_h5_path = os.path.join(self.rgb_h5_dir, videoname + '_pngs.h5')
    rgb_h5 = h5py.File(rgb_h5_path, 'r', libver='latest', swmr=True)
    rgbs = []

    rgb = []
    ####for byte in rgb_h5["pngs"][frame_indices]:
    for frame_indices in frames:
        for byte in rgb_h5["pngs"][list(frame_indices)]:
          rgb = cv2.imdecode(byte, flags=cv2.IMREAD_COLOR)
          rgb = self.rgb_transform(rgb)
          rgbs.append(rgb)
    rgbs = torch.stack(rgbs)


    depth_h5_path = os.path.join(self.depth_h5_dir, videoname + '_maskeddepth_pngs.h5')
    depth_h5 = h5py.File(depth_h5_path, 'r', libver='latest', swmr=True)
    depths = []
    
    for frame_indices in frames:
        for byte in depth_h5['pngs'][list(frame_indices)]:
          #depth = cv2.imdecode(byte, flags=cv2.IMREAD_UNCHANGED)
          #depth = cv2.imread(byte)#, cv2.IMREAD_ANYDEPTH)
          depth = np.array(byte,dtype=np.uint16)
          ##
          depth = cropND(depth, (self.side_size, self.side_size))
          depth = np.expand_dims(depth, axis=0)
          depth = depth.astype(np.float32) / 65535
          depth = torch.FloatTensor(depth)
          depths.append(depth)
    depths = torch.stack(depths)
    


    flow_h5_path = os.path.join(self.flow_h5_dir, videoname + '_3dflow.h5')
    flow_h5 = h5py.File(flow_h5_path, 'r', libver='latest', swmr=True)
    #####flows = []
    flowsActual = []
    #####
    flows = []
    #-----
    for frame_indices in frames:
      for f in flow_h5['flow'][list(frame_indices)]:
        #flow = cropND(f, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
        #flow = skimage.measure.block_reduce(f, (8,8), np.mean)
        #flow = np.transpose(f, (2,0,1))
        flow = cv2.resize(f, (self.side_size // self.patch_size, self.side_size // self.patch_size), interpolation = cv2.INTER_AREA)
        flowActual = torch.from_numpy(f)
        #flowActual = cropND(flowActual, (self.side_size, self.side_size, 3)) 
        ###flowActual = flowActual * 10
        #flowActual = self.flow_transform(flowActual)
        #flow = torch.from_numpy(f)
        #####flow = resize(f, (self.side_size // self.patch_size, self.side_size // self.patch_size))
        ##### Resize operation using interpolate on tensors
        #'''flow = np.transpose(flow, (0, 2, 1))
        #flow = F.interpolate(flow, self.side_size // self.patch_size)
        #flow = np.transpose(flow, (0, 2, 1))
        #flow = F.interpolate(flow, self.side_size // self.patch_size)'''
        ######################
        
        #####flow = np.transpose(flow, (2,0,1))
        #flowActual = np.transpose(flowActual, (2,0,1))
        #flowActual[flowActual != 0] = flowActual[flowActual != 0] + 3.1465
        #####flow = flow * 1000 # multiply 50 to "keep proper scale" according to [1]
        flowActual = resize(flowActual, (224, 224))
        flowActual = np.transpose(flowActual, (2,0,1))
        #flowActual[flowActual<0.0001] = 0
        flowActual = flowActual * 50 # multiply 50 to "keep proper scale" according to [1]
        flowsActual.append(torch.FloatTensor(flowActual))
        #####flows.append(torch.FloatTensor(flow))
        
        
        #flow = resize(f, (self.side_size // self.patch_size, self.side_size // self.patch_size))
        #flow = cropND(flow, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
        #flow = cv2.resize(flow, (self.side_size // self.patch_size, self.side_size // self.patch_size), interpolation = cv2.INTER_AREA)
        
        flow = torch.from_numpy(flow)
        #flow[flow<0.0001] = 0
        flow = flow * 50 # multiply 50 to "keep proper scale" according to [1]
        #flow[flow != 0] = flow[flow != 0] + 1.1980
        ##flow = flow + 1.1980
        flows.append(torch.FloatTensor(flow))
    #####flows = torch.stack(flows)
    flowsActual = torch.stack(flowsActual)
    flows = torch.stack(flows) 
    
 


    camera_id = int(videoname[5:8])
    replication_id = int(videoname[13:16])


    if replication_id == 1:
      if camera_id == 2:
        view_id = 1
      elif camera_id == 1:
        view_id = 5
      elif camera_id == 3:
        view_id = 3
      else:
        raise ValueError('Unexpected camera id {} from {}'.format(
                         camera_id, videoname ) )
    elif replication_id == 2:
      if camera_id == 2:
        view_id = 2
      elif camera_id == 1:
        view_id = 4
      elif camera_id == 3:
        view_id = 1
      else:
        raise ValueError('Unexpected camera id {} from {}'.format(
                         camera_id, videoname ) )
    else:
      raise ValueError('Unexpected replication id {} from {}'.format(
                       replication_id, videoname ) )

    
    if int(videoname[9:12]) in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]: #Train set
        # view_id to categorical one-hot vector
        view_id = np.eye(5, dtype='bool')[view_id-1].astype(int) # 5 for 5 views, view_id-1 for number to index
        action_label = videoname[17:].strip("0")
        # action_label to categorical one-hot vector
        #action_label = np.eye(60, dtype='bool')[int(action_label)-1].astype(int) # 60 for 60 action classes, action_label-1 for number to index
        action_label = int(action_label)
    elif int(videoname[9:12]) not in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]: #Validation set
        view_id = np.tile(np.array(view_id), (10, 1))
        action_label = int(videoname[17:].strip("0"))
        action_label = np.tile(np.array(action_label), (10,))
    
    
    action_id = int(videoname[17:20])
    setup_id = int(videoname[1:4])
    subject_id = int(videoname[9:12])

    other_cameras = [1,2,3]
    other_cameras.remove(camera_id)
    
    otherview_camera_id = other_cameras[0]
    otherview_camera_id2 = other_cameras[1]
    # Get depths images from different camera
    otherview_videoname = 'S{:03d}C{:03d}P{:03d}R{:03d}A{:03d}'.format(
      setup_id, otherview_camera_id, subject_id, replication_id, action_id)
    otherview_videoname2 = 'S{:03d}C{:03d}P{:03d}R{:03d}A{:03d}'.format(
      setup_id, otherview_camera_id2, subject_id, replication_id, action_id)
    
    otherview_videonames = []
    
    
    otherview_videonames.append(otherview_videoname)
    otherview_videonames.append(otherview_videoname2)
    
    
    otherview_length = self.meta['framelength'][otherview_videoname]
    

    otherview_frame_indices = frames
    
    otherview_depths = []
    otherview_flows = []
    for otherview_videoname_index in otherview_videonames:

        otherview_depth_h5_path = os.path.join(
          self.depth_h5_dir, otherview_videoname_index + '_maskeddepth_pngs.h5')
        otherview_depth_h5 = h5py.File(otherview_depth_h5_path, 'r', libver='latest', swmr=True)
 
        for frame_indices in otherview_frame_indices:
            for byte in otherview_depth_h5['pngs'][list(frame_indices)]:
              #depth = cv2.imdecode(byte, flags=cv2.IMREAD_UNCHANGED)
              #depth = cv2.imread(byte)#, cv2.IMREAD_ANYDEPTH)
              depth = np.array(byte,dtype=np.uint16)
              ##
              depth = cropND(depth, (self.side_size, self.side_size))
              depth = np.expand_dims(depth, axis=0)
              depth = depth.astype(np.float32) / 65535
              depth = torch.FloatTensor(depth)
              otherview_depths.append(depth)
        


        otherview_flow_h5_path = os.path.join(
          self.flow_h5_dir, otherview_videoname_index + '_3dflow.h5')
        otherview_flow_h5 = h5py.File(otherview_flow_h5_path, 'r', libver='latest', swmr=True)
        for frame_indices in otherview_frame_indices:        
            for f in otherview_flow_h5['flow'][list(frame_indices)]:
              #flow = cropND(f, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
              #flow = resize(f, (self.side_size // self.patch_size, self.side_size // self.patch_size))
              flow = cv2.resize(np.asarray(f), (self.side_size // self.patch_size, self.side_size // self.patch_size), interpolation = cv2.INTER_AREA)
              flow = np.transpose(flow, (2,0,1))
              #flow[flow<0.0001] = 0
              flow = flow * 50 # multiply 50 to "keep proper scale" according to [1]
              #flow[flow != 0] = flow[flow != 0] + 1.1193
              ##flow = flow + 1.1193
              otherview_flows.append(torch.FloatTensor(flow))
        
    
    otherview_depths = torch.stack(otherview_depths)
    otherview_flows = torch.stack(otherview_flows)


    """
    print("-------------videoname------------", videoname)
    print("-------------RGB------------", rgb.size())
    print("-------------RGBs------------", rgbs.size())
    print("-------------depths------------", depths.size())
    print("-------------flows------------", flows.size())
    print("-------------otherview_depths------------", otherview_depths[0:60,:,:,:].size())
    print("-------------otherview_flows------------", otherview_depths[0:60,:,:,:].size())
    print("-------------otherview_depths2------------", otherview_depths[60:,:,:,:].size())
    print("-------------otherview_flows2------------", otherview_depths[60:,:,:,:].size())
    print("-------------action_label------------", action_label)
    """
    
    sample = {'action_id': action_id,
              'camera_id': camera_id,
              'setup_id': setup_id,
              'subject_id': subject_id,
              'replication_id': replication_id,
              'view_id': torch.Tensor(view_id),
              'rgbs': rgbs,
              'depths': depths,
              'otherview_depths': otherview_depths[0:(self.target_length*10),:,:,:],
              'otherview_flows': otherview_flows[0:(self.target_length*10),:,:,:],
              'otherview2_depths': otherview_depths[(self.target_length*10):,:,:,:],
              'otherview2_flows': otherview_flows[(self.target_length*10):,:,:,:], 
              'action_label': action_label,
              'flows': flows,
              'flowsActual':flowsActual,
              'videoname': videoname,
              #'videoname_sequence':videoname_sequence
              }
    

    return sample
