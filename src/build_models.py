# --- Imports ---
import os
import warnings

import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from keras import models

warnings.filterwarnings("ignore")


# --- Functions ---
def classification(model, drop_out, batch_normalization):
    if drop_out is None and batch_normalization is False:
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(30, activation="softmax"))
    elif drop_out is not None and batch_normalization is False:
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(drop_out))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(drop_out))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(drop_out))
        model.add(Dense(30, activation="softmax"))
    elif drop_out is None and batch_normalization is True:
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(30, activation="softmax"))
    else:
        model = models.Sequential()
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(30, activation="softmax"))
    return model


def compile_model(model):
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["acc"])
    model.summary()
    return model


def callbacks(save_name):
    # save model and best weights
    saving_model_weights = ModelCheckpoint(save_name + ".hdf5", monitor="val_acc", verbose=0, save_best_only=True,
                                           save_weights_only=False, mode="auto", period=10)
    # reduce learning rate when close to optimum
    reduce_lr = ReduceLROnPlateau(monitor="val_acc", factor=0.1, patience=20, verbose=0, mode="auto", min_delta=0.0001,
                                  min_lr=0)
    # if NaN occurs, stop model
    nan_problem = TerminateOnNaN()
    # stop training if validation accuracy is not changing or getting worse
    early_stop = EarlyStopping(monitor="val_acc", min_delta=0, patience=20, verbose=0, mode="auto", baseline=None,
                               restore_best_weights=True)
    callbacks_list = [early_stop, nan_problem, reduce_lr, saving_model_weights]
    return callbacks_list


def get_data(training_dir, validation_dir, batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(training_dir, target_size=(128, 128), seed=340,
                                                        batch_size=batch_size)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(128, 128), seed=893,
                                                                  batch_size=batch_size)
    return train_generator, validation_generator


def fit_model(model, train_generator, validation_generator, batch_size, epochs, callbacks_list):
    history = model.fit_generator(train_generator, steps_per_epoch=2000//batch_size, epochs=epochs,
                                  validation_data=validation_generator, validation_steps=800//batch_size,
                                  callbacks=callbacks_list)
    return history


def get_results(history, save_name):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.legend()
    plt.savefig(save_name + ".png")


def finish_building_model(model, training_dir, validation_dir, save_name,
                          epochs, batch_size, drop_out, batch_normalization):
    # classification
    model = classification(model, drop_out, batch_normalization)

    # compile model
    model = compile_model(model)

    # callbacks
    callbacks_list = callbacks(save_name)

    # get data
    train_generator, validation_generator = get_data(training_dir, validation_dir, batch_size)

    # fit model
    history = fit_model(model, train_generator, validation_generator, batch_size, epochs, callbacks_list)

    # plot training accuracy vs. validation accuracy
    get_results(history, save_name)


def build_cnn_model(training_dir, validation_dir, save_name, epochs, batch_size, drop_out, batch_normalization):
    """
    Build a CNN model
    """

    # feature learning

    # save_name encoded
    # numbers of output filters     - save_name_<number>

    # 3x3 feature size
    # 32 32 64                      - none
    # 32 64 128                     - 1
    # 32 32 64 128                  - 2
    # 32 64 64 128                  - 3
    # 32 64 64                      - 6

    # 5x5 feature size
    # 32 32 64                      - 4
    # 32 64 128                     - 5

    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # finish building model
    finish_building_model(model, training_dir, validation_dir, save_name,
                          epochs, batch_size, drop_out, batch_normalization)


def build_transfer_learning_model(transfer_learning_model, training_dir, validation_dir, save_name,
                                  epochs, batch_size, unfreeze, drop_out, batch_normalization):
    """
    Build a transfer learning model
    (pre-trained on ImageNet database)
    """

    # feature learning

    # transfer learning model
    # . not including the 3 fully-connected layers at the top of the network
    tf_models = {"VGG16": VGG16, "VGG19": VGG19,
                 "ResNet50": ResNet50, "ResNet101": ResNet101, "ResNet152": ResNet152,
                 "InceptionV3": InceptionV3}
    conv_base = tf_models[transfer_learning_model](weights="imagenet", include_top=False, input_shape=(128, 128, 3))

    # transfer learning model's last 3 layers unfrozen to allow for additional training on speech database
    if unfreeze:
        for layer in conv_base.layers[:-3]:
            layer.trainable = False
    else:
        conv_base.trainable = False

    # finish building model
    model = models.Sequential()
    model.add(conv_base)
    finish_building_model(model, training_dir, validation_dir, save_name,
                          epochs, batch_size, drop_out, batch_normalization)


def build_model(data_set, save_name, transfer_learning_model=None, epochs=500, batch_size=20,
                unfreeze=False, drop_out=None, batch_normalization=False):
    """
    Build a model
    (calls either build_cnn_model or build_transfer_learning_model)

    Arguments
    ---------
    data_set:                   input image data
                                options: ["oscillograms", "spectrograms"]

    save_name:                  name under which to save best model and weights

    Optional Arguments
    ------------------
    transfer_learning_model:    transfer learning model
                                options: ["VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152", "InceptionV3"]
                                default to None

    epochs:                     number of epochs to train the model
                                default to 500

    batch_size:                 number of samples per gradient update
                                default to 20

    unfreeze:                   whether or not to unfreeze transfer learning model's last 3 layers
                                to allow for additional training on speech database,
                                only needed when transfer learning model is used
                                default to False

    drop_out:                   whether or not to apply drop out to the input
                                float between 0 and 1
                                default to None

    batch_normalization:        whether or not to apply batch normalization to the input
                                default to False
    """

    # decide where to save best model and weights
    # also, get training and validation directories
    if data_set == "spectrograms":
        os.chdir("C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\models\\spectrograms")
        training_dir = "C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\data\\processed\\spectrograms\\training"
        validation_dir = "C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\data\\processed\\spectrograms\\validation"
    else:
        os.chdir("C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\models\\oscillograms")
        training_dir = "C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\data\\processed\\oscillograms\\training"
        validation_dir = "C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\data\\processed\\oscillograms\\validation"

    # call either build_cnn_model or build_transfer_learning_model
    if transfer_learning_model is None:
        build_cnn_model(training_dir, validation_dir, save_name, epochs, batch_size, drop_out, batch_normalization)
    else:
        build_transfer_learning_model(transfer_learning_model, training_dir, validation_dir, save_name,
                                      epochs, batch_size, unfreeze, drop_out, batch_normalization)
