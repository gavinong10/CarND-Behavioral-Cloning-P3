from model import *
import pandas as pd

train_batch_size = 9000
VAL_PORTION = 0.1

template = {
    "optimizer": 'Adam',
    "lr": 0.0001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": None,
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": 64,
    "VAL_BATCH_SIZE": 64,

    "EPOCHS": 140,
    "OFFSET": 0.15,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0   
}

all_instances = [
    {**template, **{
    "optimizer": 'Nadam',
    "output_path": 'training4/Nadam_std',
}},
    {**template, **{
    "output_path": 'training4/Adam_std',
}}]

if __name__ == "__main__":
    
   # to_run = [instance1, instance2]
    to_run = all_instances
    for inst in to_run:
        pd.Series(inst).to_csv('params.csv')
        model = nvidia_model(learning_rate=inst['lr'], dropout=inst['dropout'], optimizer = inst['optimizer'])
        train(model, **inst)