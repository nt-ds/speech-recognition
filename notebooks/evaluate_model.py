# --- Import libraries ---
import os
import warnings

import numpy as np

from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")


# --- Get the current working directory ---
cwd = os.getcwd()


# --- Change directories ---
os.chdir("C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\src")


# --- More import ---
import evaluate_models


# --- Get test data ---
test_dir = "C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\data\\processed\\spectrograms\\test"
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(128, 128), seed=340, shuffle=False)


# --- Change directories ---
os.chdir("C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\models\\spectrograms")


# --- Load the best model and its weights ---
model = load_model("1_cnn_base_2.hdf5")


# --- Evaluate the best model ---
score = model.evaluate_generator(test_generator)
print("\nTest loss:      ", score[0])
print("Test accuracy:  ", score[1])
print()

label_map = test_generator.class_indices
label_map = dict((v, k) for k, v in label_map.items())
predictions = model.predict_generator(test_generator)
predictions = np.argmax(predictions, axis=-1)
predictions = [label_map[k] for k in predictions]
real_labels = test_generator.classes
real_labels = [label_map[k] for k in real_labels]

target_names = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy",
                "house", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven",
                "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero"]

evaluate_models.plot_confusion_matrix(real_labels=real_labels,
                                      predicted_labels=predictions,
                                      target_names=target_names)


# --- Change back to the original directory ---
os.chdir(cwd)
