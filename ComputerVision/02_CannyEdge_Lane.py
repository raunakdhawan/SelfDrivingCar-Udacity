import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Load the image and show
image = mpimg.imread('exit-ramp.jpg')
# plt.imshow(image)

# Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 7
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 50
high_threshold = 100
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(gray, cmap='Greys_r')
ax2.imshow(blur_gray, cmap='Greys_r')
ax3.imshow(edges, cmap='Greys_r')
plt.show()