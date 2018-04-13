from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import vgg16

def _process_data(data):
        # normalization
        data = np.clip(np.fabs(data), -np.inf, np.inf)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
def get_image_data(filename):
  img = cv2.imread(filename,1)
  img_data = np.array(img)
  return img_data

def input_data(filename, batch_size, start_pos=-1,shuffle=False):
  lines = open(filename,'r')
  read_dirnames = []
  action = []
  scene=[]
  image = []
  ac_label = []
  sc_label = []
  mc_label = []
  filename=[]
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
    
  file='./skipthoughts.npz'
  cluster=np.load(file)
  cluster_256=np.load('./clusters_256.npz')
  cluster_100=np.load('./clusters_100.npz')
  #cluster_a=list(cluster['a'])
  #cluster_s=list(cluster['s'])
  #cluster_m=list(cluster['m'])

  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
    
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split('\t')
    dirname=line[0]
    line_num = line[2]
    
    tmp_ac_label = cluster_100['ac'][int(line_num)]
    tmp_sc_label = cluster_100['sc'][int(line_num)]
    tmp_mc_label = cluster_256['mc'][int(line_num)]
    #tmp_action= cluster_a[int(line_num)]
    #tmp_scene= cluster_s[int(line_num)]
    tmp_action=np.random.rand(4800)
    tmp_scene=np.random.rand(4800)
    if not shuffle:
      print("Loading a image from {}...".format(dirname))
    
    crop_size=224
    img = get_image_data(dirname)
    img = np.array(cv2.resize(np.array(img),(crop_size,crop_size))).astype(np.float32)
    img = _process_data(img)
    
    image.append(img)
    action.append(tmp_action)
    scene.append(tmp_scene)
    ac_label.append(tmp_ac_label)
    sc_label.append(tmp_sc_label)
    mc_label.append(tmp_mc_label)
    filename.append(dirname)
    batch_index = batch_index + 1
    read_dirnames.append(dirname)
  actions = np.array(action).astype(np.float32)  
  scenes = np.array(scene).astype(np.float32)  
  images = np.array(image).astype(np.float32)
  ac_labels = np.array(ac_label).astype(np.int64)
  sc_labels = np.array(sc_label).astype(np.int64)
  mc_labels = np.array(mc_label).astype(np.int64)

  return actions,scenes,images,ac_labels,sc_labels,mc_labels, next_batch_start, read_dirnames