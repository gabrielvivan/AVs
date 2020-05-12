# **Behavioral Cloning** 

## Writeup

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model loss parameters"
[image2]: ./architecture.png "Architecture"
[image3]: ./ex_img.png "Training image"
[image4]: ./flipped.png "Flipped training Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 convolution layers using 3x3 filter sizes and depths between 32 and 128 (model.py lines 88-93), two Max Pooling layers with strides of 2x2 (lines 90 and 95), and finally one fully connected layer outputing a single steering angle for each image that is used as input (line 97).

The model includes RELU layers to introduce nonlinearity (code lines 88, 91 and 93), and the data is normalized in the model using a Keras lambda layer (86).

Initially, I had attempted to implement the same architecture as used by NVIDIA, and I also tried to transfer learning from the V3 Inception architecture. Although these models seems to operate well, it was quite challenging to fine tune it to my needs for this project. By simplifying the layers, I found it easier to fine tune, and the results were better.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 89, 92 and 94). Each of these employ a 0.2 dropout rate.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 64). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. As seen below, the training and validation losses are comparable, which show that the model is not overfitted.

![alt text][image1]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of smooth center lane driving, recovering from the left and right sides of the road multiple times. This was a key factor to success, since the vehicle would try to recover sharply as soon as it found itself driving on the sides. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to adapt the architecture employed by NVIDIA. I thought this model might be appropriate because of its depth, which would contribute to making the model more acturate.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it now included dropout layers, and MaxPooling. 

Then I noticed that the model was more complex than necessary, and it was becoming too difficult to fine tune it to my needs, since the vehicle kept going off the road in the same exact spots. I simplified the architecture by removing a couple of layers, and only leaving one fully connected layer at the top.

The final step was to run the simulator to see how well the car was driving around track one. It finally completed a full lap, with very smooth turns overall.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is the one described in the previous section, and is displayed below:

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

I recorded a large amount of training data, and uploaded the images obtained from the center camera. Then I augmented the data by creating an inverted copy of each image and multiplying the steering angle by -1. The data provided by Udacity's repo was also added to the training set. After the collection process, I had over 12,000 data points. I then preprocessed this data by cropping out the hood of the car and other portions of the image that did not display the road. The results can be seen below, for an original image and a flipped image.

![alt text][image3]
![alt text][image4]


Then I repeated this process on track two in order to get more data points. However, I chose to only use these points in the case the model needed more data. Since this was not the case, this data was just saved for later. For future work, I would implement these datapoints so that the car would also be able to autonomously drive on track 2.

Laslty, the images from the left and right camera were also saved for the case I needed to augment the data even more. Once again, this was not the case, so this strategy was not implemented.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

The model was then saved as `model.h5`, which is used by the `drive.py` script. By running this commands, the vehicle is able to autonomously drive around the track, as evidenced in the video saved as `run1.mp4`.
