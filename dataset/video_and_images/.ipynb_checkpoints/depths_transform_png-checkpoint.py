#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np

# Parameters
image_ext = '.png' # png for lossless image

def main():
  args = get_args()

  videonames = next(os.walk(args.top_dir))[1]
  videonames = [d for d in videonames if not d.startswith('.')]

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  length = len(videonames)
  videonames.sort()


  if args.num_worker < 1:
    for ind, videoname in enumerate(videonames):
      resize_images(args, ind, videoname, length)
  else:
    from joblib import Parallel, delayed
    Parallel(n_jobs=args.num_worker)(
      delayed(resize_images)(args, ind, videoname, length) for ind, videoname in enumerate(videonames)
      )

def resize_images(args, ind, videoname, length):
  print('{}/{}. Resizing images from {}...'.format(ind+1, length, videoname))
  images_dir = os.path.join(args.top_dir, videoname)
  images = os.listdir(images_dir)
  images = [f for f in images if f.endswith(image_ext) and not f.startswith('.')]
  target_dir = os.path.join(args.output_dir, videoname)
  if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
  for image in images:
    input_image_path = os.path.join(images_dir, image)
    target_img_path = os.path.join(target_dir, image)
    im = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE) # -1 for read image AS IS, e.g. 16bit monochrome
    
    
    im = cv2.resize(im, (args.target_width, args.target_height))


    im_np = im.astype(np.uint32)
    im_np = im_np*255
    
    #im_np[im_np<20] = 0
    #im_np[im_np>65535] = 65535
    #im = im_np.astype(np.uint16)
    
    im_np[im_np>255] = 255
    im = im_np.astype(np.uint8)
    
    #im_np = cv2.equalizeHist(im_np)
    
    
    im = cv2.bilateralFilter(im,4,50,50)
    
    im = cv2.medianBlur(im,3)
    im = cv2.medianBlur(im,3)
    im = cv2.medianBlur(im,3)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    im = cv2.morphologyEx(im,cv2.MORPH_OPEN,kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    
    filter = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]])
    # Applying cv2.filter2D function on our Logo image
    im=cv2.filter2D(im,-1,filter)
    
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # Applying cv2.filter2D function on our Logo image
    im=cv2.filter2D(im,-1,filter)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    

    im_np = im.astype(np.uint32)
    im_np = im_np*5000
    
    im_np[im_np<20] = 0
    
    im_np[im_np>65535] = 65535
    im_np = im_np.astype(np.uint16)
    
    
    
    cv2.imwrite(target_img_path, im_np)

def get_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

  # Input Directories and Files
  parser.add_argument('--top-dir', dest='top_dir',
    default='./dataset/NTU_RGB+D/MaskedDepthMaps/nturgb+d_depth_masked',
    help='Directory that contains directories of depth images.')

  # Output Directories
  parser.add_argument('--output-dir', dest='output_dir',
    default='./dataset/NTU_RGB+D/MaskedDepthMaps_resized/nturgb+d_depth_masked',
    help='Directory for outputs, resized pngs.')

  # Parameters
  parser.add_argument('--target-width', dest='target_width',
    type=int, default=320, help='Target width of extracted image.')
  parser.add_argument('--target-height', dest='target_height',
    type=int, default=240, help='Target height of extracted image.')

  # Parallelism
  parser.add_argument('--num-worker', dest='num_worker',
    type=int, default=0, 
    help='Number of parallel jobs for resizing. 0 for no parallelism')

  args = parser.parse_args()

  return args

if __name__ == "__main__":
  main()
