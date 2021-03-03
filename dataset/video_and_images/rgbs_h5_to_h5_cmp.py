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

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  with open(args.videoname_json_path, 'r') as fp:
    meta = json.load(fp, object_pairs_hook=collections.OrderedDict)

  videonames = meta['videonames']
  videonames = sorted(videonames)
  framecounts = meta['framelength']
  
  
  length = len(videonames)

  if args.num_worker < 1:
    for ind, videoname in enumerate(videonames):
      pngs_to_h5(args, ind, videoname, length,framecounts[videoname])
  else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.num_worker)(
      delayed(pngs_to_h5)(args, ind, videoname, length) for ind, videoname in enumerate(videonames)
      )

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since):
  now = time.time()
  s = now - since
  return '%s' % (asMinutes(s))

def pngs_to_h5(args, videoname_ind, videoname, length,frames):
  start_time = time.time()


  rgb_h5_path = os.path.join(args.top_dir, videoname + '_pngs.h5')
  rgb_h5 = h5py.File(rgb_h5_path, 'r', libver='latest', swmr=True)

  outfile_path = os.path.join(args.output_dir, videoname + '_pngs.h5')
  ########outfile_path = os.path.join(args.output_dir, videoname + '_maskeddepth_pngs.h5')  
  outfile = h5py.File(outfile_path, 'w')
  dset = outfile.create_dataset('pngs', (len(rgb_h5['pngs'][:]),), 
    maxshape=(len(rgb_h5['pngs'][:]),), chunks=True, dtype=dt,compression= "gzip")



  for byte in rgb_h5["pngs"]:
    dset[0:len(rgb_h5['pngs'][:])] = np.fromstring(byte, dtype=np.uint8)

  outfile.close()
  print('{}/{}. converting pngs of {} to h5 done...took {}'.format(
    videoname_ind+1, length, videoname, timeSince(start_time)) )

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
