import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_lib = "x"
lab_lib = "y"

num_training_images = 500
training_start_index = 1000
training_end_index = training_start_index+num_training_images

test_starting_index = 2900
num_test_images = 50
test_end_index = test_starting_index+num_test_images

img_paths = glob.glob(img_lib+"/*.png")
lab_paths = glob.glob(lab_lib+"/*.png")

img_num = len(img_paths)
lab_num = len(lab_paths)
message = "The number of images found is {0:4d} and the number of labels is {1:4d}".format(img_num, lab_num)
print(message)

#ensuring that they are in the same order:

img_paths.sort()
lab_paths.sort()

#creating the dictionary with the selected slices of paths

training_img_paths = {"x": img_paths[training_start_index:training_end_index], "y": lab_paths[training_start_index:training_end_index]}
test_img_paths = {"x": img_paths[test_starting_index:test_end_index], "y": lab_paths[test_starting_index:test_end_index]}

from loading_helpers import *
print("Image loading has started.")
ready_im, ready_lab = load_imgs_parallel(training_img_paths, num_slices=50)
print("Image loading has finsihed.")
from get_unet import *
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import losses
input_img = Input((640,640,1))
model = get_unet(input_img, n_filters = 16, dropout = 0.5, batchnorm=True)


selected_loss = losses.binary_crossentropy

model.compile("adam", loss=selected_loss, metrics=["accuracy", losses.mean_squared_error, losses.logcosh, losses.kullback_leibler_divergence,  losses.binary_crossentropy, losses.mean_squared_logarithmic_error])

print("The model has been compiled. The training will begin shortly")
print(model.summary())
model.load_weights("./ckpts/2d_sigm_norm.hdf5")

checkpoint = tf.keras.callbacks.ModelCheckpoint("./ckpts/2d_sigm_norm_2.hdf5", save_best=1, monitor="val_loss", mode="auto")
tboard = tf.keras.callbacks.TensorBoard("./ckpts/tb2/", histogram_freq=1, batch_size=10, write_graph = True, write_grads = True, write_images = True, update_freq = "batch")
callbacks = [checkpoint, tboard]

datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)#,
                               #rotation_range=15,
                               #width_shift_range=0.1,
                               #height_shift_range=0.1,
                               #shear_range=0.01,
                               #zoom_range=[0.5, 1.25],
                               #horizontal_flip=True,
                               #vertical_flip=True,
                               #fill_mode='reflect',
                               #brightness_range=[0.5, 1.5])#rotation_range=360, fill_mode = "reflect", horizontal_flip=1, vertical_flip=1,)
#datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_std_normalization= 1, rotation_range=360, fill_mode = "reflect", horizontal_flip=1, vertical_flip=1, validation_split=0.2)
#you have to use datagen.fit if u want to use featurewise_std_normalization so that the generator learns about the features

model.fit_generator(datagen.flow(x=ready_im, y=ready_lab , batch_size = 20, shuffle=True), steps_per_epoch = 10, epochs = 60,verbose = True, callbacks = callbacks, use_multiprocessing = False)