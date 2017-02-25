from model import *

train_batch_size = 9000
VAL_PORTION = 0.1

all_instances = [{
    "output_path": 'training/Adam_100_epochs_standard',

    "optimizer": 'Adam',
    "lr": 0.0001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": ['data_download'], #None
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.2,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0
},{
    "output_path": 'training/Adam_100_epochs_all_data',

    "optimizer": 'Adam',
    "lr": 0.0001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": None,
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.2,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0
},{
    "output_path": 'training/Nadam_100_epochs_standard',

    "optimizer": 'Nadam',
    "lr": 0.0001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": None,
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.2,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0
},{
    "output_path": 'training/Nadam_100_epochs_standard_lr=0.001',

    "optimizer": 'Nadam',
    "lr": 0.001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": ['data_download'],
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.2,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0
},{
    "output_path": 'training/Nadam_100_epochs_standard_dropout=0.25',

    "optimizer": 'Nadam',
    "lr": 0.0001,
    "dropout": 0.25,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": ['data_download'],
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.2,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0
},{
    "output_path": 'training/Nadam_100_epochs_standard_dropout=0.75',

    "optimizer": 'Nadam',
    "lr": 0.0001,
    "dropout": 0.75,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": ['data_download'],
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.2,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0
},{
    "output_path": 'training/Nadam_100_epochs_standard_offset=0.1',

    "optimizer": 'Nadam',
    "lr": 0.0001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": ['data_download'],
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.1,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0
},{
    "output_path": 'training/Nadam_100_epochs_standard_offset=0.3',

    "optimizer": 'Nadam',
    "lr": 0.0001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": ['data_download'],
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.3,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0
},{
    "output_path": 'training/Nadam_100_epochs_with_shadow',

    "optimizer": 'Nadam',
    "lr": 0.0001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": ['data_download'],
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.2,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": True,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.0,
    "MIN_ANGLE_VAL": 0.0
},{
    "output_path": 'training/Nadam_100_epochs_min_angle=0.05',

    "optimizer": 'Nadam',
    "lr": 0.0001,
    "dropout": 0.5,
    "min_perc_val": 0.75,
    "max_perc_val": 0.85,
    "track1_data_dirs": ['data_download'],
    "folders_to_exclude": [],
    "TRAIN_BATCH_SIZE": train_batch_size,
    "VAL_BATCH_SIZE": VAL_PORTION*train_batch_size/(1-VAL_PORTION),

    "EPOCHS": 150,
    "OFFSET": 0.2,
    "VAL_PORTION": VAL_PORTION,
    "INCLUDE_MIRROR_TRAIN": True,
    "INCLUDE_MIRROR_VAL": False,
    "INCLUDE_SHADOW_TRAIN": False,
    "INCLUDE_SHADOW_VAL": False,
    "MIN_ANGLE_TRAIN": 0.05,
    "MIN_ANGLE_VAL": 0.0
}]

if __name__ == "__main__":
    
   # to_run = [instance1, instance2]
    
    for inst in to_run:
        model = nvidia_model(learning_rate=inst['lr'], dropout=inst['dropout'], optimizer = inst['optimizer'])
        train(model, **inst)