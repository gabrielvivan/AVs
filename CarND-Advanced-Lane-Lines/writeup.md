## Writeup
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/cam_undistort.png "Undistorted"
[image2]: ./output_images/undistorted.png "Road Transformed"
[image3]: ./output_images/binary_warped.png "Binary Example"
[image4]: ./output_images/roi.png "Warp Example"
[image5]: ./output_images/sliding_window.png "Fit Visual"
[image6]: ./output_images/new_search.png "Search areas"
[image7]: ./output_images/test2_result.png "Test 2 output"
[image8]: ./output_images/test4_result.png "Test 4 output"
[image9]: ./output_images/test5_result.png "Test 5 output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first cell after defining functions in the IPython notebook located in "./P2.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After calibrating the camera, I created the function `cal_undistort` that inputs new images and uses the calibration matrix in order to undistort them. After applying this function, we get the following image:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, after using a perspective transform further described next. I found this order of steps to be relatively easier to implement, so that I could focus on the lanes and ignore the rest of the image. The thresholding steps are found in the 5th code cell of the IPython notebook.  Here's an example of my output for this step. 

![alt text][image3]

The thresholding techniques used include gradient (`Sobel()`) and HLS color change, thresholding the S channel, as seen used in function `lane_det_binary()` in the 2nd code cell of the IPython notebook.

```python
    lane_binary = lane_det_binary(bird_eye, s_thresh=(160, 240), sx_thresh=(50, 100))
```

Initially, gradient direction thresholding was implemented, but then taken out, as it did not seem to be helping much.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

As previously mentioned, I have applied a perspective transform before applying a color transform or creating a binary threshold. The code for my perspective transform includes a function called `warper()`, which appears in in the 2nd code cell of the IPython notebook.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the source and destination points by inspection in the following manner:

```python
    src = np.float32([[180,img_size[1]],[590, 450], [690, 450], [1100,img_size[1]]])
    dst = np.float32([[250,img_size[1]],[250,0],[1010,0],[1010,img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 180, 720      | 250, 720      | 
| 590, 450      | 250, 0        |
| 690, 450      | 1010, 0       |
| 1100, 720     | 1010, 720     |

As seen below, the transformed worked as expected, because the lane lines appear parallel in the transformed image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used a sliding window method to identify lane-line pixels in a test image. In this method, I first take a histogram of pixels of the bottom half image, to identify where the lanes are. Then, I specify a margin for windows to be generated surrounding the identified histogram peaks. In this case, the margin was set to 60, and the vertical number of windows was set to be 8. A minimum of number of pixels to be found is set to 50 in order to recenter the windows around the identified pixels.

Next, the function `fit_polynomial()` (found in the 2nd cell of the IPython notebook) is used to fit the pixels' position with a polynomial, as seen below: 

![alt text][image5]

After an initial polynomial is fit, I use its coefficients to guide the "search area" for new pixels. This is done by offsetting the polynomial by another margin, in this case 100. The fit polynomial function is used again, but now within this new area of search. This process is then iterated throughout the entire video.

In the images below, the new search area based on a prior polynomial is displayed, along with the detected lane.

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this inside the functions `search_around_poly()` and `measure_curvature_real()`, in the 2nd code cell of the IPython notebook. Using the conversions from pixels to meters as specified in the project, I first calculate the curvature as follows:

```python
    # Define conversions in x and y from pixels space to meters
    my = 30/720 # meters per pixel in y dimension
    mx = 3.7/700 # meters per pixel in x dimension
    
    # Define y-value where we want radius of curvature
    y_eval = 360
    
    # Implement radius of curvature calculation
    #left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    left_curverad = ((1 + (2 * mx / (my ** 2) * left_fit[0] * y_eval + mx / my * left_fit[1])**2)**(3/2)) / (np.absolute(2*mx / (my ** 2) * left_fit[0]))
    right_curverad = ((1 + (2 * mx / (my ** 2) * right_fit[0] * y_eval + mx / my * right_fit[1])**2)**(3/2)) / (np.absolute(2 * mx / (my ** 2) * right_fit[0]))
    
    return left_curverad, right_curverad
```

And then I take the average of the two calculated curvature to be the radius, while the center is just the middle point between the two fitted polynomials evaluated at the bottom of the image:

```python
    # Calculate center of lane
    center = (left_fitx[len(ploty)-1] + right_fitx[len(ploty)-1]) / 2
    
    # Compute distance from center in meters
    dist = np.absolute((binary_warped.shape[1]/2) - center) * 3.7/700
    
    # Calculate radius of curvature of lane at bottom of image
    left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit)
    radius = (left_curverad + right_curverad) / 2
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step by using the function `post_processing()`, in which an `unwarper()` function does the opposite of what the perspective transform function did on step 3. Both functions are found in the 2nd code cell of the IPython notebook, and a few test images can be seen below:

![alt text][image7]
![alt text][image8]
![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
