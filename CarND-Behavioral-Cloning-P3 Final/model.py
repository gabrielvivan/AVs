import os
import csv
import cv2
import pandas
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


def get_data():
    # This function collects the paths to the images being used for training and validating the model
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    with open('./driving_log3.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples  

def preprocess(image):
    # This function crops the images to exclude undesirable parts
    if type(None) == type(image):
      return image
    sizey = image.shape[0]    
    image = image[70:sizey-20, :]
    return image

def generator(samples, batch_size=64):
    # Generator used to process images more efficiently
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                
                center_image = preprocess(cv2.imread(name))
                fliped_image = cv2.flip(center_image, 1)
                center_angle = float(batch_sample[3])
                fliped_angle = -center_angle               
                
                images.append(center_image)
                angles.append(center_angle)
                images.append(fliped_image)
                angles.append(fliped_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Retrieve the data
samples = get_data()
X_train, X_valid = train_test_split(samples, test_size=0.2)
print('Number of training samples:' len(X_train))

# Get image to input shape of DNN & save example of preprocessed frame
name = './IMG/'+X_train[0][0].split('/')[-1]
ex_img = preprocess(cv2.imread(name))
ex_flip = cv2.flip(ex_img, 1)
cv2.imwrite('ex_img.png', ex_img)
cv2.imwrite('flipped.png', ex_flip)

# Set batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(X_train, batch_size=batch_size)
validation_generator = generator(X_valid, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Lambda, Cropping2D
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(70,320,3)))
# model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))
model.add(Conv2D(32, (3, 3), strides=(2, 2), activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), strides=(2, 2), activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(1))

# Compile using adam optimizer and fit to train/validation data
model.compile('adam', loss='mse', metrics=['accuracy'])

#add early stopping callback to stop once overfitting occurs
es = EarlyStopping(verbose=1)
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(X_train)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(X_valid)/batch_size), epochs=5, verbose=1, callbacks=[es])

# Save the model
model.save('model.h5')
print('Model saved!')

# Model results visualization
fig = plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Mean Square Error Loss')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.savefig('model.png')
print('Picture saved!')





