#!/usr/bin/env python3
import os
import h5py
import json
import time
import math
import argparse
import numpy as np
import collections

# Parameters
dt = h5py.special_dtype(vlen=np.uint8)

def main():
  args = get_args()


  # Parameters
  video_ext = '.bin'

  videos = os.listdir(args.videos_dir)
  videos = [f for f in videos if f.endswith(video_ext) and not f.startswith('.')]

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  length = len(videos)

  '''for ind, video in enumerate(videos):
    print('{}/{}. Extracting PNGs from {}...'.format(ind+1, length, video))
    video_file = os.path.join(args.videos_dir, video)
    target_dir = os.path.join(args.output_dir, os.path.splitext(video)[0])
    if not os.path.isdir(target_dir):
      os.makedirs(target_dir)
    target_img_path = os.path.join(target_dir, 'image_%06d.png')
    target_fps_cmd = ''
    if args.target_fps is not None:
      target_fps_cmd = '-r {}'.format(args.target_fps)
    cmd = 'ffmpeg -i \"{}\" -vf scale={}:{} {} -sws_flags lanczos -q:v {} \"{}\"'.\
      format(video_file, args.target_width, args.target_height, target_fps_cmd, args.target_quality, target_img_path)
    subprocess.call(cmd, shell=True)'''


  if args.num_worker < 1:
    for ind, video in enumerate(videos):
      pngs_to_h5(args, ind, video, length)
  else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.num_worker)(
      delayed(pngs_to_h5)(args, ind, video, length) for ind, video in enumerate(videos)
      )

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since):
  now = time.time()
  s = now - since
  return '%s' % (asMinutes(s))

def pngs_to_h5(args, videoname_ind, videoname, length):
  start_time = time.time()
  
  #######video_dir = os.path.join(args.top_dir, videoname + '_rgb')
  video_dir = os.path.join(args.top_dir, videoname ) #+ '_rgb')

  #files = os.listdir(video_dir)

  ########files = [os.path.join(video_dir, f) for f in files if f.startswith('image') and f.endswith('.png')]
  #files = [os.path.join(video_dir, f) for f in files if f.startswith('MDepth') and f.endswith('.png')]

  #files = sorted(files)

  total_frames = extract_depth_data(video_dir)

    
  ########outfile_path = os.path.join(args.output_dir, videoname + '_pngs.h5')
  outfile_path = os.path.join(args.output_dir, videoname + '_maskeddepth_pngs.h5')  

  outfile = h5py.File(outfile_path, 'w')
  dset = outfile.create_dataset('pngs', (len(total_frames),(240,320)), 
    maxshape=(len(files),(240,320)), chunks=True, dtype=dt)

  for f_ind, f in enumerate(files):
    # read png as binary and put into h5
    png = open(f, 'rb')
    binary_data = png.read()
    dset[f_ind] = np.fromstring(binary_data, dtype=np.uint16)
    png.close()

  outfile.close()
  print('{}/{}. converting pngs of {} to h5 done...took {}'.format(
    videoname_ind+1, length, videoname, timeSince(start_time)) )

    
    
    
    
    
    
    
    
    
    
    
    
    
def get_frame(one_complete):
    index = 0
    actual_frame = []
    for r in range(240):
        actual_frame.append( one_complete[index:index+(4*320)] )
        index = index+(5*320)
    af = [j for i in actual_frame for j in i]
    af2=[ b''.join(x) for x in zip(af[0::4], af[1::4], af[2::4], af[3::4]) ]
    arr2 = np.array( [struct.unpack('I',x) for x in af2]  ).reshape((240,320))
    return arr2

def extract_depth_data(file):
    f = open(file, 'rb')
    count = 0
    data = []
    while True:
        piece = f.read(1) 
        data.append(piece)
        if not piece:
            break
    f.close()
    
    frames = struct.unpack('i',b''.join(data[0:4]))[0]
    cols = struct.unpack('i',b''.join(data[4:8]))[0]
    rows = struct.unpack('i',b''.join(data[8:12]))[0]
    main_data = data[13:]

    total_frames = []
    actual_frame = []
    one_complete = main_data[0:384000]

    for i in range(frames):
        one_complete = data[i*384000 : (i*384000+384000)]
        actual_frame = get_frame(one_complete)
        total_frames.append(actual_frame)
        
    return total_frames
    

def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  # Input Directories and Files
  parser.add_argument('--videoname-json-path', dest='videoname_json_path',
    default='./dataset/NTU_RGB+D/ntu_rgbd_videonames.json',
    help='Path to the JSON file with videonames of NTU RGB+D dataset videos.')
  parser.add_argument('--top-dir', dest='top_dir',
    default='./dataset/NTU_RGB+D/RGBVideos/nturgb+d_rgb_png',
    help='Top Directory with subdirectories containing extracted'
         ' 3d flow results in text files.')

  # Output Directories
  parser.add_argument('--output-dir', dest='output_dir',
    default='./dataset/NTU_RGB+D/nturgb+d_rgb_pngs_320x240_lanczos_h5/',
    help='Directory for outputs, extracted 3D flows in HDF5 format.')

  # Parallelism
  parser.add_argument('--num-worker', dest='num_worker',
    type=int, default=0, 
    help='Number of parallel jobs for resizing. 0 for no parallelism.'
         ' Choose this number wisely for speed and I/O bottleneck tradeoff.')

  args = parser.parse_args()

  return args





if __name__ == "__main__":
  main()
