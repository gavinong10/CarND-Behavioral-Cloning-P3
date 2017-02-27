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
    "LOAD_MODEL": False
}

all_instances = [
    {**template, **{
    "optimizer": 'Adam',
    "lr": 0.0001,
    "output_path": 'training6_smallpreprocess/Adam_std',
    "EPOCHS": 15,
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
        print(model.summary())
        train(model, **inst)