import cv2

img = cv2.imread('image.jpg')
h, w, channels = img.shape

half = w//2

half2 = h//2

top_left = img[:half2, :] 

top_right = img[:half2, half:] 



bottom_left = img[half2:, :half]
bottom_right = img[half2:, half:]

cv2.imshow('Top Left part', top_left)
cv2.imshow('Top Right part', top_right)
cv2.imshow('Bottom Left Part', bottom_left)
cv2.imshow('Bottom Right Part', bottom_right)

cv2.imwrite('topl.jpg', top_left)
cv2.imwrite('topr.jpg', top_right)
cv2.imwrite('bottoml.jpg', bottom_left)
cv2.imwrite('bottomr.jpg', bottom_right)
cv2.waitKey(0)
