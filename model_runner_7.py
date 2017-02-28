from model import *
import pandas as pd
from keras.models import load_model
from keras.optimizers import Adam, Nadam

train_batch_size = 9000
VAL_PORTION = 0.1

template = {
    "optimizer": 'Adam',
    "lr": 0.0001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": ['data_download'],
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": 32,
    "VAL_BATCH_SIZE": 32,

    "EPOCHS": 140,
    "OFFSET": 0.15,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0 ,
    "LOAD_MODEL": False,
    "EPOCH_SIZE": None,
}

all_instances = [
# {**template, **{
#     "optimizer": 'Nadam',
#     "lr": 0.005,
#     "output_path": 'training4_high_lr/Nadam_std_+_firsthardrightfwddirt_lr0.005',
#     "track1_data_dirs": ['data_download', 'avoid_dirt', 'first_hard_right_fwd'],
#     "LOAD_MODEL": 'training4_high_lr/Nadam_std/model_epoch_17.h5',
#     "EPOCHS": 15,
# }},
# {**template, **{
#     "optimizer": 'Nadam',
#     "lr": 0.001,
#     "output_path": 'training4_high_lr/Nadam_std_+_firsthardrightfwddirt_lr0.001',
#     "track1_data_dirs": ['data_download', 'avoid_dirt', 'first_hard_right_fwd'],
#     "LOAD_MODEL": 'training4_high_lr/Nadam_std_+_firsthardrightfwddirt_lr0.005/model_epoch_15.h5',
#     "EPOCHS": 30,
# }},
{**template, **{
    "optimizer": 'Adam',
    "lr": 0.00005,
    "output_path": 'training7_0unbiased_newsmallfirsthardrightfwddirt',
    "track1_data_dirs": ['data_download', 'avoid_dirt', 'first_hard_right_fwd'],
    "EPOCHS": 15,
    "EPOCH_SIZE": 100000,
}},
{**template, **{
    "optimizer": 'Adam',
    "lr": 0.00005,
    "output_path": 'training7_0unbiased_alldata',
    "EPOCHS": 15,
    "EPOCH_SIZE": 100000,
}}]

if __name__ == "__main__":
    
   # to_run = [instance1, instance2]
    to_run = all_instances
    for inst in to_run:
        pd.Series(inst).to_csv('params.csv')
        if inst["LOAD_MODEL"]:
            model = load_model(inst["LOAD_MODEL"])
            if inst["optimizer"] == "Adam":
                opt = Adam(lr=inst["lr"])
            elif inst["optimizer"] == "Nadam":
                opt = Nadam(lr=inst["lr"])
            model.compile(optimizer=opt, loss='mean_squared_error')
        else:
            model = nvidia_model(learning_rate=inst['lr'], dropout=inst['dropout'], optimizer = inst['optimizer'])
        model.save('test.hd5')
        train(model, **inst)