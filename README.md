# Project structure
The aim of this project is to try and implement a detection algorithm to identify road features such as detecting lane boundaries and surrounding vehicles. For detecting lane boundaries, a computer vision technique library such as opencv has been used and for vehicle detection the same library with pre-trained yolo weight has been chosen to perform the algorithm.


![ezgif com-video-to-gif (2)](https://user-images.githubusercontent.com/51369142/85700210-103d5b80-b6d4-11ea-8894-d36eef4cf0d1.gif)

The pipeline to identify the road boundaries, comprises the following steps:

1. Calculate camera calibration matrix using ` cv2.findChessboardCorners()` function in order to remove the distortion generated from lenses and ensure that lane detection algorithm can be generalized to different cameras and then apply the distortion correction to the raw image.

2. Detecting the edges on the image by using set of gradient and color based threshold using `cv2.Sobel` and `cv2.cvtColor` function 
