import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import cv2
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'


# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height,3))) #32 outputs with matrix 3x3 and stride=1
model.add(Activation('relu')) #removes negatives values and puts 0 instead
model.add(MaxPooling2D(pool_size=(2, 2))) #takes the maximum between a square of 2x2

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) #transform the data to one dimension
model.add(Dense(64)) #neurons layer of 64 
model.add(Activation('relu'))
model.add(Dropout(0.5)) #prevents overfitting
model.add(Dense(1)) #output neuron
model.add(Activation('sigmoid'))#yields the probability of the class

# binary_crossentropy is the preferred loss function for binary outputs
# rmsprop is a gradient descent optimizer
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

nb_epoch = 100
nb_train_samples = 2048
nb_validation_samples = 826

"""
model.fit_generator(
        train_generator,
        steps_per_epoch=64,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples)

model.save_weights('models/basic_cnn_20_epochs.h5')
"""

model.load_weights('models/basic_cnn_20_epochs_gpu.h5')

#model.evaluate_generator(validation_generator, nb_validation_samples)
animal='/cats'
first_files=os.listdir(validation_data_dir+animal)[:400] # ls
cat=validation_data_dir+animal+'/'+first_files[251]

image = load_img(cat, target_size=(img_width, img_height))
image = img_to_array(image)

# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through the network
image = np.expand_dims(image, axis=0)

# classify the image
preds = model.predict(image)
preds = preds.tolist()[0][0]

if preds>0.5:
	print("It's a dog!")
else:
	print("It's a cat!")
print(preds)

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
orig = cv2.imread(cat)
cv2.imshow("Classification", orig)
cv2.waitKey(0)


