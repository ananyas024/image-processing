
import cv2
import numpy as np
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print("Error: Could not load image")
        exit()
    return image

def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])  # Translation matrix
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return translated_image

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)  # Rotate around the center of the image
    M = cv2.getRotationMatrix2D(center, angle, 1)  # Rotation matrix
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

def scale_image(image, scale_x, scale_y):
    scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return scaled_image

def main():
    image_path = 'image.png' 
    image = load_image(image_path)

    translated_image = translate_image(image, 50, 100)  
    cv2.imshow('Translated Image', translated_image)

    rotated_image = rotate_image(image, 45)  # Rotate by 45 degrees
    cv2.imshow('Rotated Image', rotated_image)

    scaled_image = scale_image(image, 1.5, 1.5)  # Scale by 1.5x in both x and y directions
    cv2.imshow('Scaled Image', scaled_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()