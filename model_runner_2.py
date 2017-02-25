from model import *

train_batch_size = 9000
VAL_PORTION = 0.1

template = {
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
}

all_instances = [
#     {**template, **{
#     "output_path": 'training/Nadam_std',
# }}, 
#      {**template, **{
#     "output_path": 'training/Nadam_0.3_dropout',
#     "dropout": 0.3
# }},
#     {**template, **{
#     "output_path": 'training/Nadam_0.6_dropout',
#     "dropout": 0.6
# }},
    {**template, **{
    "output_path": 'training/Nadam_0.9_dropout',
    "dropout": 0.9
}},
    {**template, **{
    "output_path": 'training/Nadam_offset=0.1',
    "OFFSET": 0.1
}},
    {**template, **{
    "output_path": 'training/Nadam_offset=0.15',
    "OFFSET": 0.15
}},
    {**template, **{
    "output_path": 'training/Nadam_offset=0.5',
    "OFFSET": 0.25
}}]

if __name__ == "__main__":
    
   # to_run = [instance1, instance2]
    to_run = all_instances
    for inst in to_run:
        model = nvidia_model(learning_rate=inst['lr'], dropout=inst['dropout'], optimizer = inst['optimizer'])
        train(model, **inst)