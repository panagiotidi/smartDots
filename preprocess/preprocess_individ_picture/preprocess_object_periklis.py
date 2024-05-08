import cv2

import config

# path = '/Users/sofia/PycharmProjects/smartDots/data/original_02_04_2024/Ammodytes_0_2018_195_11ce9871-8774-439a-9e41-e6cb0a93160b.jpg'
path = '/data/original_02_04_2024/Lepidorhombus whiffiagonis_15_2022_410_a6ddd476-de0d-4d3c-85f2-6ac91c309417.jpg'


def periklis_preprocess(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    featuresList = []
    featuresAreaList = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        feature = image[y:y+h, x:x+w]
        featuresList.append(feature)
        featuresAreaList.append(cv2.contourArea(contour))

    sorted_featuresAreaList = sorted(featuresAreaList, reverse=True)
    index_of_highest = featuresAreaList.index(sorted_featuresAreaList[0])
    # index_of_second_highest = featuresAreaList.index(sorted_featuresAreaList[1])

    # areasDif = featuresAreaList[index_of_highest]/featuresAreaList[index_of_second_highest]

    # if areasDif < 20:
    #     cv2.imwrite('./lepfeature_1.jpg', featuresList[index_of_highest])
    #     cv2.imwrite('./lepfeature_2.jpg', featuresList[index_of_second_highest])
    # else:
    #     cv2.imwrite('./lepfeature_1.jpg', featuresList[index_of_highest])
    #
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return_img = cv2.resize(featuresList[index_of_highest], (config.INPUT_WIDTH, config.INPUT_HEIGHT))

    return return_img


if __name__ == '__main__':

    image = cv2.imread(path)
    periklis_preprocess(image)
    # Save or display the results as needed
    cv2.imwrite('./lepresult2.jpg', image)