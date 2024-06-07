import cv2
import numpy as np

import config

# path = '/Users/sofia/PycharmProjects/smartDots/data/original_02_04_2024/Ammodytes_0_2018_195_11ce9871-8774-439a-9e41-e6cb0a93160b.jpg'
path = '/Users/sofia/PycharmProjects/smartDots/data/original_06_05_2024/Gadus morhua/0A7ABB41-F348-4406-BB3D-8A6686D0D7A2.png_0.png'


def pre_process_sofia2(image):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the black background)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), thickness=cv2.FILLED)

    # Bitwise AND the original image with the mask
    result = cv2.bitwise_and(image, mask)

    # Find bounding box coordinates of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding box coordinates
    cropped = result[y:y + h, x:x + w]

    # Check size and rotate to always have width > height
    height, width, channels = cropped.shape
    if height > width:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    return cropped


if __name__ == '__main__':

    image = cv2.imread(path)
    image = pre_process_sofia2(image)
    # Save or display the results as needed
    cv2.imwrite('./output_sofia2.jpg', image)