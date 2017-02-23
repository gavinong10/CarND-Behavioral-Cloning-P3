import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import random
from keras.models import load_model, Sequential, Model
from keras.layers import Cropping2D
import cv2
import os
import socket
import scipy
from sklearn import preprocessing
import shutil
import skimage.transform as sktransform

is_AWS = False if 'Macbook' in socket.gethostname() else True

if is_AWS:
    track1_dir = '/home/carnd/Dropbox/udacity-data/track1'
else:
    track1_dir = '/Users/macbook/Development/personal/udacity-car/CarND-Behavioral-Cloning-P3/track1'

try:
    folders_to_exclude = pd.read_csv('to_exclude.csv', header=None, names=['Index', 'Name'])['Name'].tolist()
    model = load_model('model.h5')
except:
    folders_to_exclude = []
    model = None
    
folders_to_exclude += ['.DS_Store']

track1_data_dirs = [x for x in os.listdir(track1_dir) if x not in folders_to_exclude]
print(track1_data_dirs)

#############
track1_data_dirs = ['data_download']
#############

track1_data_dirs = [track1_dir + '/' + x for x in track1_data_dirs]

driving_log_df = None

for data_dir in track1_data_dirs:
    df = pd.read_csv(data_dir + "/driving_log.csv", header=None, names=["center","left","right","steering","throttle","brake","speed"])

    cols = ['center', 'left', 'right']
    for col in cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.split("/").apply(lambda x: x[-1])
    df[['center', 'left', 'right']] = data_dir + "/IMG/" + df[['center', 'left', 'right']]
    
    if driving_log_df is None:
        driving_log_df = df
    else:
        driving_log_df = pd.concat([driving_log_df, df])

driving_log_df.to_csv('driving_log.csv', index=False)