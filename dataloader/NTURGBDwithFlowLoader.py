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

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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

        
    for videoname in self.meta['videonames']:
      if self.subset == 'all':
        self.videonames.append(videoname)
        
        length = self.meta['framelength'][videoname]-6
        #seq_list = list (range(1, (length-(length%self.target_length))//self.target_length+1))
        seq_size = (length-(length%self.target_length))//self.target_length
        for f in range(1,seq_size+1):
            self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))

        
      elif self.subset == 'train':
        if int(videoname[9:12]) in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]:
            # Cross-Subject Evalutation subjects (perferomers id), 40,320 samples
            self.videonames.append(videoname)
            
            length = self.meta['framelength'][videoname]-6
            
            #seq_list = list (range(1, (length-(length%self.target_length))//self.target_length+1))
            seq_size = (length-(length%self.target_length))//self.target_length
            for f in range(1,seq_size+1):
                self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))

      else: # test set
        if int(videoname[9:12]) not in [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]:
            # 16,560 samples
            self.videonames.append(videoname)
            
            length = self.meta['framelength'][videoname]-6
            #seq_list = list (range(1, (length-(length%self.target_length))//self.target_length+1))
            seq_size = (length-(length%self.target_length))//self.target_length
            for f in range(1,seq_size+1):
                self.videonames_sequences.append(videoname+"F"+'{0:03d}'.format(f))
    
    self.videonames = sorted(self.videonames)
    self.videonames_sequences = sorted(self.videonames_sequences)
    
    
    #print("total videos------------------------",len(self.videonames))
    #print("total new stuff------------------------",len(self.videonames_sequences))
    #exit()
    
    if self.visual_transform is None:
      self.rgb_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.CenterCrop(self.side_size),
                              transforms.ToTensor()
                              ])
    elif self.visual_transform == 'normalize':
      self.rgb_transform = transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.CenterCrop(self.side_size),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                              ])

    ###print('NTURGBDwithFlow is initialized. '
    ###      '(Subset: {}, Length: {})'.format(self.subset, len(self.videonames_sequences)) )

    print('NTURGBDwithFlow is initialized. '
          '(Subset: {}, Length: {})'.format(self.subset, len(self.videonames_sequences)) )
    
  def __len__(self):
    return len(self.videonames_sequences)
    #return len(self.videonames)

  def __getitem__(self, idx):
    
    ####print ('-------------------------------------------',idx)
    
    
    videoname_sequence = self.videonames_sequences[idx]
    videoname = videoname_sequence[0:20]
    
    #print("videoname------------------------",videoname)
    #print("videoname_seq--------------------",videoname_sequence)
    
    frame_segment = int(videoname_sequence[21:])
    #print("frame_segment--------------------",str(frame_segment))
    
    #########print("total___frame--------------------",self.meta['framelength'][videoname])
    #########print("framesssssssssss--------------------", list(range( (frame_segment*self.target_length)-self.target_length, (frame_segment*self.target_length))) )
    

    
    ####print("*************Videoname**********",videoname)
    ####print([db_item == videoname for db_item in self.meta['framelength']])
    ####print("!!!!!!!!!!!!!!!!!!!self meta size: ",self.meta['framelength'][videoname])
    
    
    length = self.meta['framelength'][videoname]
    
    ##cropped_length = (length - (length % self.target_length) )
    ##frame_indices = list (range(cropped_length))
    
    frame_indices = list(range( (frame_segment*self.target_length)-self.target_length, (frame_segment*self.target_length)))
    
    
    #print("=======actual={}/{}==={}".format(frame_indices,length,videoname) )

    #####frame_indices = np.linspace(0, cropped_length-1, endpoint=False, 
    #####                            num=cropped_length-1, dtype=int)
      # Exclude last frame so we can use the same number of flow images
    #####frame_indices = list(frame_indices)

    
    

    rgb_h5_path = os.path.join(self.rgb_h5_dir, videoname + '_pngs.h5')
    rgb_h5 = h5py.File(rgb_h5_path, 'r')
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
    depth_h5 = h5py.File(depth_h5_path, 'r')
    depths = []
    
    #print("-------------RGB------------", rgb.size())
    #print("-------------RGBs------------", rgbs.size())
    #print("-------------RGBs 0------------", len(rgbs[:]))
    #print("-----------frames--------------", frame_indices)



    for byte in depth_h5['pngs'][frame_indices]:
      depth = cv2.imdecode(byte, flags=cv2.IMREAD_UNCHANGED)
      depth = cropND(depth, (self.side_size, self.side_size))
      depth = np.expand_dims(depth, axis=0)
      depth = depth.astype(np.float32) / 65535
      depth = torch.FloatTensor(depth)
      depths.append(depth)
    depths = torch.stack(depths)
    
    
    #print("-------------depths------------", depths.size())


    flow_h5_path = os.path.join(self.flow_h5_dir, videoname + '_3dflow.h5')
    flow_h5 = h5py.File(flow_h5_path, 'r')
    flows = []
    for f in flow_h5['flow'][frame_indices]:
      flow = cropND(f, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
      flow = np.transpose(flow, (2,0,1))
      flow = flow * 50 # multiply 50 to "keep proper scale" according to [1]
      flows.append(torch.FloatTensor(flow))
    flows = torch.stack(flows)
    
    #print("-------------flows------------", flows.size())

    #############################################print("=================actual={}/{}==={}===".format(frame_indices,length,videoname),idx )
    
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
    
    #print("-------otherview_length--------",otherview_length)
    
    #if np.abs(length - otherview_length) >= 5 :
    #    print("length=",length,"- otherview=",otherview_length,"- video=",videoname, "- other-video=",otherview_videoname)
    #    print("frame_indices=", frame_indices)
    
    otherview_frame_indices = frame_indices
    
    otherview_depths = []
    otherview_flows = []
    for otherview_videoname_index in otherview_videonames:

        otherview_depth_h5_path = os.path.join(
          self.depth_h5_dir, otherview_videoname_index + '_maskeddepth_pngs.h5')
        otherview_depth_h5 = h5py.File(otherview_depth_h5_path, 'r')
 
        for byte in otherview_depth_h5['pngs'][otherview_frame_indices]:
          depth = cv2.imdecode(byte, flags=cv2.IMREAD_UNCHANGED)
          depth = cropND(depth, (self.side_size, self.side_size))
          depth = np.expand_dims(depth, axis=0)
          depth = depth.astype(np.float32) / 65535
          depth = torch.FloatTensor(depth)
          otherview_depths.append(depth)
        

        #print("-------------otherview_depths------------", otherview_depths.size())   


        otherview_flow_h5_path = os.path.join(
          self.flow_h5_dir, otherview_videoname_index + '_3dflow.h5')
        otherview_flow_h5 = h5py.File(otherview_flow_h5_path, 'r')
        for f in otherview_flow_h5['flow'][otherview_frame_indices]:
          flow = cropND(f, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
          flow = np.transpose(flow, (2,0,1))
          flow = flow * 50 # multiply 50 to "keep proper scale" according to [1]
          otherview_flows.append(torch.FloatTensor(flow))
        
    
    otherview_depths = torch.stack(otherview_depths)
    otherview_flows = torch.stack(otherview_flows)

    flow_h5_path = os.path.join(self.flow_h5_dir, videoname + '_3dflow.h5')
    flow_h5 = h5py.File(flow_h5_path, 'r')
    flows = []
    for f in flow_h5['flow'][frame_indices]:
      flow = cropND(f, (self.side_size // self.patch_size, self.side_size // self.patch_size, 3)) # centercrop
      flow = np.transpose(flow, (2,0,1))
      flow = flow * 50 # multiply 50 to "keep proper scale" according to [1]
      flows.append(torch.FloatTensor(flow))
    flows = torch.stack(flows)


    
    
    ##print("-------------otherview_depths------------", otherview_depths.size())   
    ##print("-------------otherview_flows------------", otherview_flows.size())
    ##print("-------------otherview_depths------------", otherview_depths[0:6,:,:,:].size())   
    ##print("-------------otherview_flows------------", otherview_flows[6:,:,:,:].size())
    ##exit()

    #print("-------view_id--------",view_id)
    #print("-------torch.Tensor(view_id)--------",torch.Tensor(view_id))
    sample = {'action_id': action_id,
              'camera_id': camera_id,
              'setup_id': setup_id,
              'subject_id': subject_id,
              'replication_id': replication_id,
              'view_id': torch.Tensor(view_id),
              'rgbs': rgbs,
              'depths': depths,
              'otherview_depths': otherview_depths[0:6,:,:,:],
              'otherview_flows': otherview_flows[0:6,:,:,:],
              'otherview2_depths': otherview_depths[6:,:,:,:],
              'otherview2_flows': otherview_flows[6:,:,:,:],             
              'flows': flows,
              'videoname': videoname
              }
    

    return sample

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def NTURGBDwithFlowLoader(
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
