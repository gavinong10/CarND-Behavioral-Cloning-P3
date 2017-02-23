import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import os
import socket
from sklearn import preprocessing
import skimage.transform as sktransform
import preprocess

#############
#track1_data_dirs = ['data_download']
#############


def return_df_with_processed_paths(data_dir):
    df = pd.read_csv(data_dir + "/driving_log.csv", header=None, names=["center","left","right","steering","throttle","brake","speed"])

    cols = ['center', 'left', 'right']
    for col in cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.split("/").apply(lambda x: x[-1])
    df[['center', 'left', 'right']] = data_dir + "/IMG/" + df[['center', 'left', 'right']]
    return df

def concat_df(df, df1):
    if df is None:
        return df1
    return pd.concat([df, df1])

def return_track1_dataframe(track1_data_dirs = None, folders_to_exclude = []):
    is_AWS = False if 'Macbook' in socket.gethostname() else True

    if is_AWS:
        track1_dir = '/home/carnd/Dropbox/udacity-data/track1'
    else:
        track1_dir = '/Users/macbook/Development/personal/udacity-car/CarND-Behavioral-Cloning-P3/track1'
    
    if track1_data_dirs is None:
        folders_to_exclude += ['.DS_Store']        
        track1_data_dirs = [x for x in os.listdir(track1_dir) if x not in folders_to_exclude]
    
    track1_data_dirs = [track1_dir + '/' + x for x in track1_data_dirs]

    driving_log_df = None

    for data_dir in track1_data_dirs:
        df = return_df_with_processed_paths(data_dir)
        
        driving_log_df = concat_df(driving_log_df, df)

    return driving_log_df

def organize_data(df):
    df_left = df.copy()
    df_right = df.copy()
    
    df_left['position'] = 'left'
    df_right['position'] = 'right'
    df['position'] = 'center'
    
    df = pd.concat([df, df_left, df_right])
    df['path'] = df[df['position']]
    del(df[['left', 'center', 'right']])
    df = df.sample(frac=1).reset_index(drop=True)
    return df
    
def subset_by_mode(df, mode):
    total_len = len(df)
    val_len = int(val_portion * total_len)
    train_len = total_len - val_len
    
    if mode == "train":
        df = df[:train_len]
    else: #Validation set
        df = df[train_len:]
    
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


def retrieve_generator(df, image_size, batch_size = 10, mode = 'train', position = 'center',
                     offset = 0.2, val_portion = 0.2, include_mirror=True, 
                     include_shadow = True, min_angle=0.02):
    df = subset_by_mode(df, mode)
    if mode == 'val':
        position = 'center'
    
    df = set_position_targets(df, position)
    df = offset_steering(df, offset)
    df = filter_by_steering(df, min_angle)

    df['mirror'] = False
    if include_mirror:
        df = append_mirrored_data(df)
        
    df['shadow'] = False
    if include_shadow:
        df = append_shadowed_data(df)

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
                
                img = preprocess_image(img)
                    
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


