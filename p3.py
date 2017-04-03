import csv
import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import math

#capture info in driving_log.csv
#this saves training data from driving_log.csv into a python variable
lines = []
with open('../training//driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images_all = []
measurements = []

steering_correction = 0.2  # TODO update steering measurement for left and right images
ec2_image_folder = '../training/IMG/'  #TODO update to actual location of images when on ec2
for line in lines:
	source_path = line[0]
	current_path_center = ec2_image_folder + line[0].split('/')[-1]
	current_path_left = ec2_image_folder + line[1].split('/')[-1]
	current_path_right = ec2_image_folder + line[2].split('/')[-1]

	measurement = float(line[3])
	measurement_center = measurement
	measurement_left = measurement + steering_correction
	measurement_right = measurement - steering_correction

	images_all.append((current_path_center, measurement_center))
	images_all.append((current_path_left, measurement_left))
	images_all.append((current_path_right, measurement_right))


def generator(samples, batch_size=100):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                angle = batch_sample[1]
                image = cv2.imread(batch_sample[0])
                #crop
                image = image[70:140]
                image = cv2.resize(image,(64,64))
                #convert to RGB
                image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #yield sklearn.utils.shuffle(X_train, y_train)
            yield(X_train, y_train)

#shuffle data 
images_all = sklearn.utils.shuffle(images_all)

#split data into training and validation group
train_samples = images_all[:int(len(images_all)*0.8)]   #get the first 80% of data
train_samples_len = len(train_samples)
validation_samples = images_all[int(len(images_all)*0.8):]   #get the last 20% of data
validation_samples_len = len(validation_samples)

batch_size = 100
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# Nvidia model

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64,64,3)))
model.add(Conv2D(24,(1,1), activation="relu"))
model.add(Conv2D(36,(2,2), activation="relu"))
model.add(Conv2D(48,(2,2), activation="relu"))
model.add(Conv2D(64,(1,1), activation="relu"))
model.add(Conv2D(64,(1,1), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
	#steps_per_epoch=32,
	samples_per_epoch=train_samples_len,
	validation_data=validation_generator,
	nb_val_samples=validation_samples_len,
	nb_epoch=4)

model.save('model.h5')
