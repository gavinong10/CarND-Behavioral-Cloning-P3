from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from preprocess import *
from data import *

def nvidia_model(learning_rate=0.0001, dropout=0.5):
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3),
                            activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))

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

    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model

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

def train(model):
    df = return_track1_dataframe(track1_data_dirs = ['data_download'])
    df = organize_data(df)

    # train model
    EPOCHS = 70
    OFFSET = 0.2 # 0.08-0.25 from forums. 4 degrees = 0.16
    VAL_PORTION = 0.1

    INCLUDE_MIRROR_TRAIN = True
    INCLUDE_MIRROR_VAL = False
    
    INCLUDE_SHADOW_TRAIN = True
    INCLUDE_SHADOW_VAL = False
    
    MIN_ANGLE_TRAIN=0.0 #1.0
    MIN_ANGLE_VAL=0.0

    IMAGE_SIZE = (160, 320, 3)
    # Train on all the data
    position = 'all'
    train_generator_all, train_len = retrieve_generator(df, IMAGE_SIZE, 100, 'train', position, 
                                                      OFFSET, VAL_PORTION, INCLUDE_MIRROR_TRAIN, 
                                                      INCLUDE_SHADOW_TRAIN, MIN_ANGLE_TRAIN)

    validation_generator_all, val_len = retrieve_generator(df, IMAGE_SIZE, 100, 'val', position, 
                                                      OFFSET, VAL_PORTION, INCLUDE_MIRROR_VAL, 
                                                      INCLUDE_SHADOW_VAL, MIN_ANGLE_VAL)


    model.fit_generator(train_generator_all,
                        samples_per_epoch=train_len,
                        nb_epoch=EPOCHS,
                        validation_data=validation_generator_all, 
                        nb_val_samples=val_len,
                        callbacks=[weight_logger()], # Add a callbacks to save best model in each epoch, but it is not neccesary
                        verbose=1)  # If verbose=1 or none, will show processbar, keep it if run without GPU


model = nvidia_model()
print(model.summary())
train(model)
model.save('model.h5')