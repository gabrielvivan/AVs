# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/vizualization.png "Visualization"
[image2]: ./examples/preprocessed.png "Preprocessing"
[image3]: ./examples/CNN.png "LeNet architecture"
[image4]: ./examples/CNN2.png "AlexNet architecture"
[image5]: ./examples/signs.png "Traffic Signs"
[image6]: ./examples/STOP_sign.jpg "Traffic Sign 2"
[image7]: ./examples/turn_right.jpg "Traffic Sign 3"
[image8]: ./examples/road_work.jpg "Traffic Sign 4"
[image9]: ./examples/yield.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/gabrielvivan/AVs/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used basic python functions to assess the size of each of the datasets used in this project. In order to count for the number of unique classes in the labels dataset, I found the maximum classID value and added 1 to it, to account for the 0th classID. Therefore:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First I pick a random image from the training dataset to display. We can see that the image is correctly imported and displayed, and that the dimensions check. Next, I plotted the histograms of the classes for the training, validation, and test labels datasets. As seen, the initial classes have many more images than the later classes. This is not a problem as long as the same pattern is observed for the validation dataset, which is the case. However, it is possible that the CNN will have more accurate guesses for the classes that had more training data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because in this way the machine does not take the sign's color as a feature for classification. This means that the machine will be more robust for detecting signs under different lighting conditions.

Then, I normalized the image data because it is important that the inputs to neural nets have zero mean and variance for better results.

As a last step, I resized the images to a shape of 32x32x1, since the color conversion operation with openCV got rid of the depth dimension.

Here is an example of a preprocessed image:

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		      |     Description	        					                 | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 Grayscale image   							             | 
| Convolution 7x7     	 | 1x1 stride, valid padding, outputs 26x26x13 	 |
| RELU and dropout			   |	Dropout probability of 0.6											         |
| Max pooling	         	| 2x2 stride,  outputs 13x13x13 				            |
| Convolution 4x4	      | 1x1 stride, valid padding, outputs 10x10x26   |
| RELU and dropout			   |	Dropout probability of 0.6											         |
| Max pooling	         	| 2x2 stride,  outputs 5x5x26 				              | 
| Fully connected		     | Inputs 650, outputs 300        									      |
| Fully connected		     | Inputs 300, outputs 86        								       	|
| Fully connected		     | Inputs 86, outputs 43        									        |
| Softmax 				          | Followed by minimizing cross-entropy       			|

The following images are diagrams of the architecture:

![alt text][image3] ![alt text][image4]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, which minimized the cross-entropy of the softmax output. The batch size was 128, the number of epochs was 10, and the learning rate was 0.0008.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.942 
* test set accuracy of 0.915

I started with the same architecture as the well known LeNet structure. This was intended to reduce the amount of hyperparameters available for tuning, and due to the similarities of the inputs of this project to the LeNet classifier. I believed it would be relevant to the traffic sign application because the inputs and features to be detected in this case are similar to the ones while detecting numbers. 

Then, I had to use more filters for each operation, therefore making the network a little deeper, with a higher number of parameters. I also introduced dropout to avoid overfitting. Then, I reduced the learning rate , as the initial results were achieving a steady accuracy level at early epochs. By doing so, the learning curve was not as steep, but the model kept a steady learning curve up until epoch 10, when it achieved 0.94 validation accuracy. The test accuracy reflects that the model is not overfitted, and it is satisfactory in classifying traffic signs.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, after being resized for the neural network:

![alt text][image5]

The first image might be difficult to classify because there are a few other signs in the dataset with very similar characteristics.
The stop and yield signs should be quite simple to classify, due to its very unique shapes.
The road work might be one of the most difficult, as many signs have the same shape, with different features inside the "triangle".
Finally, the turn right only should be relatively simple, with a possibility for mistakes with similar signs, such as "right turn or go ahead".

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			            |     Prediction	        					            | 
|:---------------------:|:---------------------------------------------:| 
| End of all limits     | End of all limits  									          | 
| Stop     			        | Stop 										                      |
| Right turn				    | Right turn											              |
| Road work	      		  | Road work					 				                    |
| Yield			            | Yield      							                      |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares well to the accuracy of the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the last cell of the Ipython notebook.

For the first image, the model is not very sure that this is a "end of passing and speed limits (probability of 0.54), but it still guessed it correctly. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .54         			| End of all speed and passing limits   									| 
| .23     				| End of no passing 										|
| .17					| End of speed limit (80km/h)											|
| .01	      			| Children crossing					 				|
| .01				    | End of no passing by vehicles over 3.5 metric tons      							|

For the second image, the model was very certain it was a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .80         			| Stop   									| 
| .09     				| Keep right 										|
| .05					| Turn left ahead											|
| .01	      			| No entry					 				|
| .01				    | Priority road      							|

The model was very confident about guessing the third and fifth images, as expected due to their distinct features. The top five were, respectively

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .89         			| Turn right ahead   									| 
| .03     				| Keep left										|
| .01					| Go straight or left											|
| .01	      			| General caution					 				|
| .01				    | Yield      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Yield   									| 
| .01     				| Ahead only 										|
| .01					| Right-of-way at the next intersection											|
| .005	      			| Go straight or left					 				|
| .003				    | Priority road      							|

Finally, as expected, the network struggled to accurately guess the road work sign, with a confidence of only 29%. However, it was still able to guess it correctly, which confirms the architecture is robust.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .29         			| Road work   									| 
| .21     				| Wild animals crossing										|
| .09					| Dangerous curve to the left											|
| .09	      			| Double curve					 				|
| .03				    | Turn right ahead      							|

