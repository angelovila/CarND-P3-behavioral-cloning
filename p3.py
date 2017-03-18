import csv
import cv2
import numpy as np


lines = []

##TODOdouble check location of file when uploaded to ec2
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		#this saves training data from file.csv into a python variable

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../data/IMG/' + filename    ##TODO update to actual location of images when on ec2
	image = cv2.imread(current_path) #opens the image, into a np array
	images.append(image) #saves an image np array into images list
	#below line gets the steering measurement from the driving_log.csv
	#steering data is the 4th data in the csv, therefore line[3]
	measurement = float(line[3])
	

	measurements.append(measurement)

############ flat model  to start#########

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizers='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
