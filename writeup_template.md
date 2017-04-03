#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* p3.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 video recording of autonomous mode going one lap

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Python file containing the model is p3.py

Model initially normalize the data using a lambda layer (p3: line) 
Following Nvidia's self driving car model, It uses 5x5 convolutions with a 2x2 strides and then couple of 3x3 convolutions, all using RELU activation.


####2. Attempts to reduce overfitting in the model

The model was trained and tested and did not appear to be overfitting and therefore no dropout layer is inserted.

The model was trained using data on a different track to help in generalization.


####3. Model parameter tuning

The model used an adam optimizer. (p3.py line).

Batch size of the generator doesn't seem to affect the model as much and sticked with 100

Epochs was kept at a 4, further increasing seem to prolong the training time with minimal increase in accuracy.

Steering correction used was 0.2. The recovery training data seem to not let the model be confused with straight or 0 angle steering.

####4. Appropriate training data

Training data consist of the following strategies:
-couple of laps with the intention of staying centered on the road on both tracks in both clockwise and counter-clockwise directions
-lap recovering from the side of the road to center
--both strategies are done in both clockwise and counter-clokwise


###Model Architecture and Training Strategy

####1. Solution Design Approach

Project was started by generating an augmented image and angle measurement of the center images. This was implemented as I thought everything starts with the data.

I did however encountered a MemoryError when I start to test my working model and start to gather more training data. Turns out that my initial implementation of image augmentation has to be set as a generator in order for it to be successfully ran. Issue is, the way my code for augmentation  was structured in a way that  it would be complicated as I have my images (already read by cv2.imread) and angle measurement in different lists.


I found out that the fit_generator is easier to setup if the image and the angle are in a tuple. It would also make sense to only start reading the image in the generator instead of a memory intesnsive and storing everything on a list. Upon reading the image, I only added two pre-processing steps which are cropping and resizing.

Cropping and resizing was later updated to be a lot smaller as it turns out, model works even with smaller images. Making the training run faster.

After making a running model, upon testing, encountered an issue where there are parts of of the track where the car run off. I tried to get more training but still failed. Turns out that drive.py input image needs to be resized as well


In summary:
-pre-process the image by cropping and resizing
-store the data to have a list of tuples containing images file location and steering angle
-since fit_generator doesn't divide by validation automatically, I shuffled then seperated my training and validation data first before running it in the generator
-use a generator when training a model
-use nvidia's model in keras
-resize input image of drive.py


####2. Final Model Architecture

Input of 64,64,3
Normalization layer
Convolution(24,(5,5) with 2x2 strides)
Convolution(36,(5,5) with 2x2 strides)
Convolution(48,(5,5) with 2x2 strides)
Convolution(64,(3,3))
Flatten layer
Fully Connected (100)
Fully Connected (50)
Fully Connected (10)
Fully Connected (1)


####3. Creation of the Training Set & Training Process

Training data was collected by running two laps on the track and then two laps running the opposite directions.

On the 2nd track, collected data by running one lap and then another lap running the opposite direction.

Then, collected data by running a recovery lap.It is done by running at the side of the road and moving to the center of the road, recording only the process of centering to the road. This was exhaustively done.

Two issues seem to happen which were:
1. Car seem to still un-recover and goes off the track
2. there are parts on the track where car goes off the track.

To remedy the issue
1. C more data by doing more recovery laps in a higher angle. Instead of going to the side of the road and then centering, recovery data is collected by going more like an angle closer to perpendicular to the side of the road and then steering to be back to the center of the road.
2. Recorded more data to the problem turns of the track.


Images recorded at the side of the car is used by manipulating the steering angle recorded to the particular instand by 0.2.

All images have the horizon and the hood of the car cropped and resized to 64x64. Cropping helps the model avoid noise and resizing to a smaller image makes the model run significantly faster


Data was shuffled prior to being fed into the generator. After shuffling data was divided into training and validation group.
