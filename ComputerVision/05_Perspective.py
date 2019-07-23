import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "./05_Perspective/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('./05_Perspective/test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
     dst = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
     dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
     ret, corners = cv2.findChessboardCorners(dst_gray, (8, 6), None)
    # 4) If corners found: 
     if ret == True:
          # If we found corners, draw them! (just for fun)
          cv2.drawChessboardCorners(dst, (nx, ny), corners, ret)
          # Choose offset from image corners to plot detected corners
          # This should be chosen to present the result at the proper aspect ratio
          # My choice of 100 pixels is not exact, but close enough for our purpose here
          offset = 100 # offset for dst points
          # Grab the image shape
          img_size = (dst_gray.shape[1], dst_gray.shape[0])

          # For source points I'm grabbing the outer four detected corners
          src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
          # For destination points, I'm arbitrarily choosing some points to be
          # a nice fit for displaying our warped result 
          # again, not exact, but close enough for our purposes
          dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                        [img_size[0]-offset, img_size[1]-offset], 
                                        [offset, img_size[1]-offset]])
          # Given src and dst points, calculate the perspective transform matrix
          M = cv2.getPerspectiveTransform(src, dst)
          # Warp the image using OpenCV warpPerspective()
          warped = cv2.warpPerspective(dst_gray, M, img_size)
     return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()