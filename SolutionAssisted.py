
# coding: utf-8

# In[1]:

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


# In[2]:

is_AWS = False if 'Macbook' in socket.gethostname() else True


# In[3]:

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import Callback

def Model(weights_path=None):
    model = Sequential()
    #Normalize to be between -1 and 1
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(32, 128, 3)))
    
    model.add(Convolution2D(16, 3, 3, input_shape=(32, 128, 3))) #(30, 126, 16)
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2))) #(15, 68, 16)
    
    model.add(Convolution2D(32, 3, 3)) #(13, 66, 32)
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2))) #(6, 33, 32)
    
    model.add(Convolution2D(64, 3, 3)) #(4, 31, 64)
    model.add(MaxPooling2D(pool_size=(2, 2))) #(2, 15, 64)
    
    model.add(Flatten()) #1920
    model.add(Dense(500)) #500
    model.add(ELU())
    model.add(Dropout(.5))
    
    model.add(Dense(100)) #100
    model.add(ELU())
    model.add(Dropout(.25))
    
    model.add(Dense(20))
    model.add(ELU())
    
    model.add(Dense(1)) # TODO: Try with Tanh

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model


# In[4]:

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

print("Length: ", len(driving_log_df))


driving_log_df.head()


# In[5]:

def get_next_image_generator(df, position = 'center', offset = 0.2):
    for idx, image_path in enumerate(df[position]):
        img = cv2.imread(image_path)
        yield img


# In[6]:

tempgen = get_next_image_generator(driving_log_df)
sample = next(tempgen)
print("Dimension of image: H x W X D = ", sample.shape)
print("# of images: ", len(driving_log_df))

print("Steering range: Min=", np.min(driving_log_df['steering']), " , Max=", np.max(driving_log_df['steering']))
print("Throttle range: Min=", np.min(driving_log_df['throttle']), " , Max=", np.max(driving_log_df['throttle']))
print("Brake range: Min=", np.min(driving_log_df['brake']), " , Max=", np.max(driving_log_df['brake']))
print("Speed range: Min=", np.min(driving_log_df['speed']), " , Max=", np.max(driving_log_df['speed']))

print("image Min: ", np.min(sample))
print("image Max: ", np.max(sample))
#sample


# In[10]:

def preprocess(image, top_offset=.375, bottom_offset=.125):
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = sktransform.resize(image[top:-bottom, :], (32, 128, 3))
    return image

def add_random_shadow(image):
    ### Add random shadow as a vertical slice of image
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image
    
def adjust_brightness(): #TODO
    pass

def offset_steering(df, offset):
    df[df['target'] == 'left']['steering'] = df[df['target'] == 'left']['steering'] + offset
    df[df['target'] == 'right']['steering'] = df[df['target'] == 'right']['steering'] - offset
    return df

def filter_by_steering(df, min_angle):
    return df[np.abs(df['steering']) >= min_angle]

def append_mirrored_data(df):
    dfMirror = df.copy(deep=True)
    dfMirror['mirror'] = True
    dfMirror['steering'] *= -1
    df = pd.concat([df, dfMirror])
    return df

def append_shadowed_data(df):
    dfShadow = df.copy(deep=True)
    dfShadow['shadow'] = True
    df = pd.concat([df, dfShadow])
    return df


def append_v_shift_noise_data(df):
    dfVShift = df.copy(deep=True)
    dfVShift['v_shift_noise'] = True
    df = pd.concat([df, dfVShift])
    return df
        
def set_position_targets(df, position):
    if position == 'all':
        dfLeft = df.copy(deep=True)
        dfLeft['target'] = 'left'
        dfCenter = df.copy(deep=True)
        dfCenter['target'] = 'center'
        dfRight = df.copy(deep=True)
        dfRight['target'] = 'right'
        df = pd.concat([dfLeft, dfCenter, dfRight])
    else:
        df['target'] = position 
    return df
    
    
# DF Columns: steering, mirror, <position>, target, type
# type is the kind of augmentation performed: center, left, right, vshifted, shadowed, brightnened

def get_next_feature(df, batch_size = 10, mode = 'train', position = 'center', 
                     offset = 0.2, val_portion = 0.2, include_mirror=True, 
                     include_shadow = True, include_v_shift_noise=True, min_angle=0.02):
    total_len = len(df)
    val_len = int(val_portion * total_len)
    train_len = total_len - val_len

    if mode == "train":
        df = df[:train_len]
    else: #Validation set
        df = df[train_len:]
        position = 'center' #Only use center data
    
    df = set_position_targets(df, position)
    df = offset_steering(df, offset)
    df = filter_by_steering(df, min_angle)
            
    df['mirror'] = False
    if include_mirror:
        df = append_mirrored_data(df)
        
    df['shadow'] = False
    if include_shadow:
        df = append_shadowed_data(df)
    
    df['v_shift_noise'] = False
    if include_v_shift_noise:
        df = append_v_shift_noise_data(df)
    
    image_size = (32, 128, 3)

    inputs = np.zeros([batch_size, *image_size]) #length of prediction output
    targets = np.zeros([batch_size])
    
    
    def generator(df, inputs, targets):
        count = 0
        while(True):
            #Shuffle
            df = df.sample(frac=1).reset_index(drop=True)
            for idx in range(len(df)):
                row = df.iloc[idx]
                image_path = row[row['target']]
                img = cv2.imread(image_path)
                if row['mirror']:
                    img = img[:,::-1,:]
                
                if row['shadow']:
                    img = add_random_shadow(img)
                
                v_delta = .05 if row['v_shift_noise'] else 0
                img = preprocess(img,
                   top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
                   bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta))
                    
                img = img[np.newaxis, :, :, :]

                inputs[count] = img
                targets[count] = row['steering']

                count += 1
                if count == batch_size:
                    yield inputs, targets
                    inputs = np.zeros([batch_size, *image_size])
                    targets = np.zeros([batch_size])
                    count = 0
    return generator(df, inputs, targets), len(df)
                


# In[11]:

# Callbacks function in model, use for save best model in each epoch, but it is not neccesary
class weight_logger(Callback):
    def __init__(self):
        super(weight_logger, self).__init__()
        # Create the weight path as empty
        self.weight_path = os.path.join('weights/')
        shutil.rmtree(self.weight_path, ignore_errors=True)
        os.makedirs(self.weight_path, exist_ok=True)
    def on_epoch_end(self, epoch, logs={}):
        #At end of epoch, save the model
        self.model.save_weights(os.path.join(self.weight_path, 'model_epoch_{}.h5'.format(epoch + 1)))

# Define the model

#model = Model(dropout=0.7, dropout_level=1, orig = False, discrete=False)
if model is None:
    model = Model()


# In[12]:

model.compile(optimizer='Nadam', loss='mean_squared_error', lr=0.0001)

# train model
EPOCHS = 20
OFFSET = 0.2 # 0.08-0.25 from forums. 4 degrees = 0.16
VAL_PORTION = 0.1

INCLUDE_MIRROR_TRAIN = True
INCLUDE_MIRROR_VAL = False
INCLUDE_SHADOW_TRAIN = True
INCLUDE_SHADOW_VAL = False
INCLUDE_V_SHIFT_NOISE_TRAIN = True
INCLUDE_V_SHIFT_NOISE_VAL = False
MIN_ANGLE_TRAIN=0.02 #1.0
MIN_ANGLE_VAL=0.0

# Train on all the data
position = 'all'
train_generator_all, train_len = get_next_feature(driving_log_df, 10, 'train', position, 
                                                  OFFSET, VAL_PORTION, INCLUDE_MIRROR_TRAIN, 
                                                  INCLUDE_SHADOW_TRAIN, INCLUDE_V_SHIFT_NOISE_TRAIN,
                                                  MIN_ANGLE_TRAIN)

validation_generator_all, val_len = get_next_feature(driving_log_df, 10, 'val', position, 
                                                  OFFSET, VAL_PORTION, INCLUDE_MIRROR_VAL, 
                                                  INCLUDE_SHADOW_VAL, INCLUDE_V_SHIFT_NOISE_VAL,
                                                  MIN_ANGLE_VAL)


model.fit_generator(train_generator_all,
                        samples_per_epoch=train_len,
                        nb_epoch=EPOCHS,
                        validation_data=validation_generator_all, 
                        nb_val_samples=val_len,
                        callbacks=[weight_logger()], # Add a callbacks to save best model in each epoch, but it is not neccesary
                        verbose=1)  # If verbose=1 or none, will show processbar, keep it if run without GPU




# In[ ]:




# In[ ]:

model.save('modeltest.h5')


# In[ ]:

#model.save('model.h5')


# In[ ]:

#Series([x.split("/")[-1] for x in track1_data_dirs], name="done_to_exclude").to_csv("to_exclude.csv")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



