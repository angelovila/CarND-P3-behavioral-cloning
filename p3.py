import csv
import cv2
import numpy as np


lines = []

##TODOdouble check location of file when uploaded to ec2
with open('../training//driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		#this saves training data from file.csv into a python variable

images_center = []
images_left = []
images_right = []
images_all = []
measurements = []
measurements_left = []
measurements_right = []
measurements_center = []
measurements_all = []
steering_correction = 0.2  # TODO update steering measurement for left and right images
ec2_image_folder = '../training/IMG/'  #TODO update to actual location of images when on ec2
for line in lines:
	source_path = line[0]
	""" before generator
	#get filenames of center, left, right images
	filename_center_image = source_path.split('/')[-1]
	filename_left_image = line[1].split('/')[-1]
	filename_right_image = line[2].split('/')[-1]

	#update to correspond to ec2 locations
	current_path_center = ec2_image_folder + filename_center_image
	current_path_left = ec2_image_folder + filename_left_image
	current_path_right = ec2_image_folder + filename_right_image

	#save images to a list
	image_center = cv2.imread(current_path_center) #opens the image, into a np array
	images_center.append(image_center) #saves an image np array into images list
	#below line gets the steering measurement from the driving_log.csv
	#steering data is the 4th data in the csv, therefore line[3]
	image_left = cv2.imread(current_path_left)
	images_left.append(image_center)
	image_right = cv2.imread(current_path_right)
	images_right.append(image_center)

	#get steering angle, with steering correction for left and right images
	measurement = float(line[3])
	measurements_center.append(measurement)
	measurements_left.append(measurement + steering_correction)
	measurements_right.append(measurement - steering_correction)

	"""
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

""" before generator
###TODO make images and measurements into numpy array
images_center = np.array(images_center)
measurements_center = np.array(measurements_center)


####### copy and flip center_images
augmented_images = []
augmented_measurements = []

for image, measurement in zip(images_center, measurements_center):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

augmented_images  = np.array(augmented_images)   #only contains center images
augmented_measurements = np.array(augmented_measurements)   #only contains center images
images_left = np.array(images_left)
images_right = np.array(images_right)

##append all images into one list using generator
def generator_append_images():
  global augmented_images,images_left, images_right
  x = 0
  while x <len(augmented_images):
    images_all.append(augmented_images[x])
    x += 1
    yield
  x = 0
  while x < len(images_left):
    images_all.append(images_left[x])
    x += 1
    yield
  x = 0
  while x < len(images_right):
    images_all.append(images_right[x])
    x += 1
    yield

images_generator = generator_append_images()
for i in range(len(augmented_images) + len(images_left) + len(images_right)):
  next(images_generator)

#X_train = np.array(X_train)  #throws a memory error
y_train = np.array(measurements_all)





images_all = np.concatenate((augmented_images, images_left, images_right), axis=0)
measurements_all = np.concatenate((augmented_measurements, measurements_left, measurements_right), axis=0)


X_train = np.array(images_all)
y_train = np.array(measurements_all)
"""




#####TODO GENERATOR FUNCTION #######
import sklearn
#shuffle samples first before running in generator
#use sklearn.utils.shuffle(samples)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                angle = batch_sample[1]
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


#shuffle data first
sklearn.utils.shuffle(images_all)

#split data into training and validation group
train_samples = images_all[len(:images_all)*0.8]   #get the first 80% of data
validation_samples = images_all[len(images_all)*0.8:]   #get the last 20% of data

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


############ flat model  to start#########

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


#################### basic network############
"""
model = Sequential()
#model.add(Lambda(lambda x: x/ 255.0 - 0.5, input shape=(160,320,3)))
#model.add(Flatten())  # no need to add input shape if first layer is preprocessing (normalize, mean center)
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
"""

#########################################
##################LeNet##################
"""
model = Sequential()

###cropping###
#model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3))) #use cropped images
#model.add(Lambda(lambda x: x/255.0 - 0.5))  #remove line below to use cropped images
############## remove line lambda to use cropping #####

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

"""


#####################################
#################Nvidia model########

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3))
model.add(Lambda(lambda x: x / 255.0 - 0.5)
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))




#### extras for reference
#model.add(Dropout(0.5))

###################
model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)  #before generator

""" example fit generator
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)
"""


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
	steps_per_epoch=32,
	samples_per_epoch=len(train_samples),
	validation_data=validation_generator,
	nb_val_samples=len(validation_samples),
	nb_epoch=4)

model.save('model.h5')
