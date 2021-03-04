#!/usr/bin/env python3
import os
import cv2
import argparse
import numpy as np

# Parameters
image_ext = '.png' # png for lossless image

def main():
  args = get_args()

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  length = 1
  ind=0
  #videonames.sort()

  resize_images(args, ind, args.top_dir, length)


def resize_images(args, ind, images_dir, length):
  print('{}/{}. Resizing images...'.format(ind+1, length))
  #images_dir = os.path.join(args.top_dir, videoname)
  images = os.listdir(images_dir)
  images = [f for f in images if f.endswith(image_ext) and not f.startswith('.')]
  target_dir = args.output_dir
  if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
  for image in images:
    input_image_path = os.path.join(images_dir, image)
    target_img_path = os.path.join(target_dir, image)
    im = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE) # -1 for read image AS IS, e.g. 16bit monochrome
    
    
    im = cv2.resize(im, (args.target_width, args.target_height))


    ##im_np = im.astype(np.uint32)
    ###im_np = im_np
    
    ##im_np[im_np>255] = 255
    #im = im.astype(np.uint8)
    
    #im_np = cv2.equalizeHist(im_np)
    
    
    im = im.astype(np.uint32)
    im = im*2000
    im[im>65535] = 65535
    im = im.astype(np.uint16)
    
    #im = cv2.bilateralFilter(im,4,50,50)
    
    im = cv2.medianBlur(im,3)
    im = cv2.medianBlur(im,3)
    im = cv2.medianBlur(im,3)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    im = cv2.morphologyEx(im,cv2.MORPH_OPEN,kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    
    
    im[im<20] = 0
    
    im[im>65535] = 65535
    im = im.astype(np.uint16)
    
    
    cv2.imwrite(target_img_path, im)

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
