import sys
import cv2
import numpy as np
from cv2.saliency import Saliency
from sklearn.linear_model import LogisticRegression

from classifier import config


np.set_printoptions(threshold=sys.maxsize)

# file = '/Users/sofia/PycharmProjects/smartDots/data/original_all_accordance/2018_195_2c99947d-a34e-4c8e-b0f6-b52bdfa06b5a.jpg'
# file = '/Users/sofia/Desktop/2018_195_2c99947d-a34e-4c8e-b0f6-b52bdfa06b5a.jpg' # 5 years
file = '/Users/sofia/Desktop/2022_412_e4d7d015-f0d8-4c2b-93da-7e359b327a65.jpg' # 5 years
# file = '/Users/sofia/Desktop/2023_523_fd7edccc-7590-43e5-aecb-b2cb3f288d6c.jpg' # 1 year
# file = '/Users/sofia/Desktop/2023_523_20591985-f948-4e16-835c-065a84ec5629.jpg' # 4 years


def get_largest_object(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    # print('out:', output)
    # print('stats:', stats, stats.shape)
    # sizes = stats[:, -1]
    # print('sizes:', sizes.shape)
    # print('max stats:', max_label, max_size)
    # Find the largest non background component.
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
    # print(centroids.shape)
    # print('max_label:', max_label)
    # print('center of largest object:', centroids[max_label])
    # print('stats:', stats[max_label]) # left, top, width, height, stat_area

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2.astype('uint8'), stats[max_label]


def mask_largest_object(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('1. Gray image', gray_img)

    lower = np.array([0])
    upper = np.array([50])
    in_range = cv2.inRange(gray_img, lower, upper)
    in_range = cv2.bitwise_not(in_range)
    # cv2.imshow('2. Gray image inRange', in_range)

    largest_mask, stats = get_largest_object(in_range)
    return largest_mask, stats


def mark_largest(image):
    largest_object_mask, stats = mask_largest_object(image)
    # cv2.imshow('3. Largest mask', largest_object_mask)

    # change mask to a 3 channel image
    # apply mask
    src1_mask = cv2.cvtColor(largest_object_mask, cv2.COLOR_GRAY2BGR)
    mask_out = cv2.subtract(src1_mask, image)
    mask_out = cv2.subtract(src1_mask, mask_out)

    return mask_out, stats


def pre_process(file, image):

    # Find and crop area of interest
    marked, stats = mark_largest(image)
    # cv2.imshow('6. Final_image', marked)
    # cv2.waitKey(0)

    # initialize OpenCV's static saliency spectral residual detector and
    # compute the saliency map
    # saliency:Saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    # (success, saliencyMap) = saliency.computeSaliency(marked)
    # saliencyMap = (saliencyMap * 255).astype("uint8")
    # cv2.imshow("Image", marked)
    # cv2.imshow("Output", saliencyMap)
    # cv2.waitKey(0)

    # print('This marked image is:', type(marked), ' with dimensions:', marked.shape)

    marked_and_cropped = marked[stats[1]: stats[1] + stats[3], stats[0]:stats[0] + stats[2]]  # Slicing to crop the image
    image_shape = marked_and_cropped.shape
    if image_shape[1] > image_shape[0]:
        marked_and_cropped = cv2.rotate(marked_and_cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # img_gray = cv2.cvtColor(marked_and_cropped, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('Original', marked_and_cropped)
    # edges = cv2.Canny(img_gray, 100, 220)

    # sobelx = cv2.GaussianBlur(sobelx, (3, 3), 0)
    # ret, thresh = cv2.threshold(cropped, 130, 255, cv2.THRESH_BINARY)
    # img_gray = cv2.bitwise_not(img_gray)
    # cv2.imshow('Grey channels only', img_gray)

    # Sobel Edge Detection

    grey = cv2.cvtColor(marked_and_cropped, cv2.COLOR_BGR2GRAY)
    # cv2.imshow(file + '1', grey)

    # sobelx = cv2.Sobel(src=grey, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)  # Sobel Edge Detection on the X axis
    # cv2.imshow('sobelx1', sobelx)
    #
    # edges = cv2.Canny(np.uint8(sobelx), 100, 200)
    # cv2.imshow('edges2', edges)
    # contours, hierarchy = cv2.findContours(grey, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # contour_image = np.zeros_like(marked_and_cropped)
    # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
    # cv2.imshow(file, contour_image)
    # cv2.waitKey(0)

    # cv2.imshow(file, sobelx)
    sobely = cv2.Sobel(src=grey, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    # cv2.imshow('6. Contour detection using grey channels only1', sobely)
    # sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # cv2.imshow('7. Contour detection using grey channels only2', sobelxy)

    # contours0, hierarchy0 = cv2.findContours(image=img_gray, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_KCOS)
    # r = cv2.drawContours(image=img_gray, contours=contours0, contourIdx=-1, color=(0, 255, 0), thickness=22, lineType=cv2.LINE_AA)
    # print(contours0)
    # cv2.imshow('Contour detection using grey channels only', r)

    return_img = cv2.resize(sobely, (config.INPUT_WIDTH, config.INPUT_HEIGHT))

    # change mask to a 3 channel image
    return_img = cv2.merge((return_img, return_img, return_img))
    return return_img


if __name__ == '__main__':
    image = cv2.imread(file)
    # print('This image is:', type(image), ' with dimensions:', image.shape)
    # cv2.imshow('0. Original image', image)
    final_image = pre_process(file, image)
    # cv2.imshow('6. Final_image', final_image)
    # cv2.waitKey(0)


