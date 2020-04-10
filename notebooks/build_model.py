# --- Import libraries ---
import os
import warnings

warnings.filterwarnings("ignore")


# --- Get the current working directory ---
cwd = os.getcwd()


# --- Change directories ---
os.chdir("C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\src")


# --- More import ---
import build_models


# --- Build model ---
data_set = "oscillograms"
save_name = "1_vgg19_base"
transfer_learning_model = "VGG19"
epochs = 500
batch_size = 20
unfreeze = False
drop_out = None
batch_normalization = False

build_models.build_model(data_set, save_name, transfer_learning_model, epochs,
                         batch_size, unfreeze, drop_out, batch_normalization)


# --- Change back to the original directory ---
os.chdir(cwd)
