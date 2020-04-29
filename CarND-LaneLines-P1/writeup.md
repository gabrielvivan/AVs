# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals of this project are the following:
* Make a pipeline that finds lane lines on the road
* Further develop pipeline so that the detected lanes are averaged and extrapolated, in order to have one continuous line representing the lane.


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg "Fig. 1"
[image2]: ./test_images_output/solidWhiteRight.jpg "Fig. 2"
[image3]: ./test_images_output/solidYellowCurve.jpg "Fig. 3"
[image4]: ./test_images_output/solidYellowLeft.jpg "Fig. 4"
[image5]: ./test_images_output/whiteCarLaneSwitch_out.jpg "Fig. 5"

---

### Reflection

### 1. Pipeline methodology

My pipeline consisted of 6 steps. First, I converted the images to grayscale, then I applied Gaussian blur with kernel size of 5. Next, function Canny was applied, with a low and high thresholds of 40 and 120, respectively. The vertices for the region of interest were then specified, and the image was masked. Then, a Hough transform was applied to detect lines, with the following parameters:

`rho = 2  (distance resolution in pixels of the Hough grid)

theta = pi/180 (angular resolution in radians of the Hough grid)

threshold = 50 (minimum number intersections in Hough grid cell)

min_line_length = 15 (minimum number of pixels making up a line)

max_line_gap = 5 (maximum gap in pixels between connectable line segments)`

Finally, the detected lines were drawn on the original image using the cv2.addWeighted function.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first converting the output of the Hough Lines transform into a 2D numpy array. Then, the x and y coordinates of the lines were separated into two arrays, by evaluating the slope of the detected lines. By doing so, I was able to calculate the average slope of the right and left lanes, and estimate the intercept. Therefore, I had all parameters that were needed to draw a line, using y = m*x + b, where y was the top and bottom values of the region of interest, allowing to solve for x.

The pipeline was evaluated using 5 test images, as seen:

![Fig. 1][image1]
![Fig. 2][image2]
![Fig. 3][image3]
![Fig. 4][image4]
![Fig. 5][image5]

Finally, the pipeline was tested in two video files, which can be found in the folder /test_videos_output.

### 2. Identify potential shortcomings with your current pipeline

As seen, the continous lines drawn in the video are not very stable. This might be due to the detection of a few horizontal lines, which interefere with the slope average used to extrapolate the lines. One potential shortcoming would be what would happen when the car is travelling in roads with many cracks. It is possible that the cracks would be detected as lanes, and decrease the slope averages. 

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to implement a condition to ignore slopes that are too different than the majority of the slopes of lines that were detected. Another improvement could be to better fine tune the hough line and canny function parameters, to more carefully detect only what is desired (i.e. no cracks, short lanes, etc.).

