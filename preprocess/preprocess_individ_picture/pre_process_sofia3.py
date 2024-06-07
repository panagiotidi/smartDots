import cv2
import numpy as np
from cv2.typing import Size

import config
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

# path = '/Users/sofia/PycharmProjects/smartDots/data/original_02_04_2024/Ammodytes_0_2018_195_11ce9871-8774-439a-9e41-e6cb0a93160b.jpg'
path = '/Users/sofia/PycharmProjects/smartDots/data/original_06_05_2024/Gadus morhua/0A7ABB41-F348-4406-BB3D-8A6686D0D7A2.png_0.png'


def crop_specific_area(image):
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
    return cropped


def increase_brightness(img, value=30):
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.split(img)[0]

    lim = 255 - value
    img[img > lim] = 255
    img[img <= lim] += value

    # final_hsv = cv2.merge((h, s, v))
    # img = cv2.cvtColor(v, cv2.COLOR_HSV2BGR)
    return img


def pre_process_sofia3(image):

    cropped = crop_specific_area(image)

    # Check size and rotate to always have width > height
    height, width, channels = cropped.shape
    if height > width:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    # sobelxy = cv2.GaussianBlur(cropped, (13, 13), cv2.BORDER_DEFAULT);
    # sobelxy = cv2.bitwise_not(cropped)

    # sobelxy = cv2.Sobel(src=sobelxy, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=7, scale=1.0)
    # grad_y = cv2.Sobel(src=sobelxy, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3, scale=1.0)
    # abs_grad_x = cv2.convertScaleAbs(grad_x);
    # abs_grad_y = cv2.convertScaleAbs(grad_y);
    # sobelxy = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0);

    # sobelxy = cv2.Laplacian(sobelxy, cv2.CV_64F)

    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # cropped = increase_brightness(cropped, value=60)

    # return cropped
    return cv2.merge((cropped, cropped, cropped))


if __name__ == '__main__':
    image = cv2.imread(path)
    image = pre_process_sofia3(image)
    # Save or display the results as needed
    cv2.imwrite('./output_sofia3.jpg', image)
