import numpy as np
import cv2

# Load the image we need 
original_image = cv2.imread('img1.jpg')
height = original_image.shape[0]
width = original_image.shape[1]
channels = original_image.shape[2]

# Convert RGB space to LAB color space
LAB = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)

# define the output as the same size of the original image
output = np.zeros((height,width,channels), dtype = np.uint8)

# Implement meanshift to the image
# The second parameter is the spatial window size
# The third parameter is the color window size
meanshift = cv2.pyrMeanShiftFiltering(LAB, 10, 50, output,1)


# Convert RGB space to LAB color space
output = cv2.cvtColor(output, cv2.COLOR_LAB2RGB)
# Write the output
cv2.imwrite('img1_out.png', output)

# Show the image
cv2.imshow('out',output)
cv2.waitKey(0)


