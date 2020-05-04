## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./output_images/test2_result.jpg)

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product is a detailed writeup of the project.  Please refer to writeup.md for a detailed step-by-step description of how this algorithm works.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing the pipeline on single frames.  

Examples of the output from each stage of the pipeline are save in the folder called `output_images`. The video called `project_video.mp4` is the video with which the pipeline was tested on. The final output of the pipeline is saved in the folder `output_images` and is named `project_video_output.avi`
