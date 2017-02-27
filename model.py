from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam, Nadam
from keras.callbacks import Callback
from preprocess import *
from data import *
import pandas as pd
from pandas import DataFrame, Series
import shutil

def nvidia_model(learning_rate=0.0001, dropout=0.5, optimizer = 'Adam'):
    model = Sequential()

    # model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3),
    #                         activation='relu'))
    model.add(Convolution2D(24, 5, 5, subsample=(1, 1), border_mode='same', input_shape=(16, 40, 3),
                            activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(1, 1), border_mode='same', activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(1, 1), activation='relu'))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    
    if optimizer == 'Adam':
        opt = Adam(lr=learning_rate)
    elif optimizer == 'Nadam':
        opt = Nadam(lr=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model

# Callbacks function in model, use for save best model in each epoch, but it is not neccesary
class weight_logger(Callback):
# class weight_logger():
    def __init__(self, model, output_path, df_download, min_perc_val=0.75, max_perc_val=0.85):
        super(weight_logger, self).__init__()
        # Create the weight path as empty
        self.output_path = os.path.join(output_path + '/')
        # Clear output path directory
        shutil.rmtree(self.output_path, ignore_errors=True)
        os.makedirs(self.output_path, exist_ok=True)
        
        self.df_download = df_download
        self.min_perc_val = min_perc_val
        self.max_perc_val = max_perc_val
        self.model = model
        
    def on_epoch_end(self, epoch, logs={}):
        #At end of epoch, save the model
        # if epoch % 100 != 0: #Only on every 100th epoch do something
        #     return
        print("\n\n", logs, "\n\n")
        
        self.model.save(os.path.join(self.output_path, 'model_epoch_{}.h5'.format(epoch + 1)))
        
        # Want to save training loss and validation loss
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        # Calculate a specialized kind of validation loss:
        # Take the original data_download data and find the images equating to the min_perc_val to max_perc_val'th percentile.
        # Calculate the predictions and average loss for these images.
        df = self.df_download
        
        min_quant = df['steering'].quantile(self.min_perc_val)
        max_quant = df['steering'].quantile(self.max_perc_val)
        
        df = df[(df['steering'] >= min_quant) & (df['steering'] <= max_quant)]
        
        def calc_pred(image_path):
            img = cv2.imread(image_path)
            img = preprocess_image(img)
            img = img[np.newaxis, :, :, :]
            steering_angle = float(self.model.predict(img, batch_size=1))
            return steering_angle
        
        df['prediction'] = df['center'].apply(calc_pred)
        df['prediction_loss'] = df['prediction'] - df['steering']
        
        MSE = (df['prediction_loss'] * df['prediction_loss']).sum() / len(df)
        
        # Repeat the above, but do so only with the averages of segments continuous < 0.1 difference;
        # then, average these segments.
        
        size_filter = df['steering'].abs().diff() < 0.1
        df['groups'] = (~size_filter).cumsum()
        MSE_segments = df[size_filter][['prediction_loss', 'groups']].groupby('groups')['prediction_loss'] \
                .aggregate(lambda x: (x * x).sum() / len(x)).mean()
        
        output_dict = {
            'train_loss' : loss,
            'val_loss' : val_loss,
            'adjusted MSE': MSE,
            'adjusted MSE on segments': MSE_segments
        }
        
        pd.Series(output_dict).to_csv(os.path.join(self.output_path, 'stats_epoch_{}.csv'.format(epoch + 1)))
        
class Namespace(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

def dump_into_namespace(**kw):
    return Namespace(**kw)

def train_on_batch(model, train_generator, train_batch_size, train_data_len, val_generator, val_data_len, epochs, weight_logger):
        points_processed = 0
        epochs_processed = 0
        for batch in train_generator:
            batch_training_loss = model.train_on_batch(batch[0], batch[1])
            points_processed += train_batch_size
            print("(Epoch ", epochs_processed, "Batch training loss: ", batch_training_loss, ";\t ", points_processed, " / ", train_data_len)
            if points_processed >= train_data_len:
                points_processed -= train_data_len
                epochs_processed += 1
                # Calculate validation loss
                val_loss = val_on_batch(model, val_generator, val_data_len, epochs)
                weight_logger.on_epoch_end(epochs_processed - 1, {'loss': batch_training_loss, 'val_loss': val_loss})
                if epochs_processed == epochs:
                    return
            
def val_on_batch(model, val_generator, val_data_len, epochs):
        epoch_val_loss = model.evaluate_generator(val_generator, val_data_len)
        return epoch_val_loss

def train(model, **kwargs):
    ns = Namespace(**kwargs)
    # Used for validation purposes
    df_download = return_track1_dataframe(track1_data_dirs = ['data_download'])

    df = return_track1_dataframe(track1_data_dirs = ns.track1_data_dirs, folders_to_exclude = ns.folders_to_exclude)
    df = organize_data(df)

    IMAGE_SIZE = (160, 320, 3)
    # Train on all the data
    position = 'all'
    train_generator_all, train_len = retrieve_generator(df, IMAGE_SIZE, ns.TRAIN_BATCH_SIZE, 'train', position, 
                                                      ns.OFFSET, ns.VAL_PORTION, ns.INCLUDE_MIRROR_TRAIN, 
                                                      ns.INCLUDE_SHADOW_TRAIN, ns.MIN_ANGLE_TRAIN)

    validation_generator_all, val_len = retrieve_generator(df, IMAGE_SIZE, ns.VAL_BATCH_SIZE, 'val', position, 
                                                      ns.OFFSET, ns.VAL_PORTION, ns.INCLUDE_MIRROR_VAL, 
                                                      ns.INCLUDE_SHADOW_VAL, ns.MIN_ANGLE_VAL)
    


    weight_logger_inst = weight_logger(model, ns.output_path, df_download)
    train_on_batch(model, train_generator_all, ns.TRAIN_BATCH_SIZE, train_len, \
                   validation_generator_all, val_len, ns.EPOCHS, weight_logger_inst)


    # model.fit_generator(train_generator_all,
    #                     samples_per_epoch=ns.TRAIN_BATCH_SIZE, #train_len,
    #                     nb_epoch=ns.EPOCHS,
    #                     validation_data=validation_generator_all, 
    #                     nb_val_samples=ns.VAL_BATCH_SIZE, #val_len,
    #                     callbacks=[weight_logger(ns.output_path, df_download)],
    #                     verbose=1)  # If verbose=1 or none, will show processbar, keep it if run without GPU

if __name__ == "__main__":
    model = nvidia_model()
    print(model.summary())
    train(model)
    model.save('model.h5')