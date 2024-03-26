import numpy as np
from scipy import ndimage

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M ,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180


    for i in range(1 , M -1):
        for j in range(1 , N -1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i ,j] < 22.5) or (157.5 <= angle[i ,j] <= 180):
                    q = img[i, j+ 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


if __name__ == '__main__':
    file = '/Users/sofia/PycharmProjects/smartDots/data/original_all_accordance/2018_195_2c99947d-a34e-4c8e-b0f6-b52bdfa06b5a.jpg'

    import numpy as np
    from PIL import Image

    # Load the image
    img = Image.open(file)
    # Preprocess the image
    img = img.resize((572, 572))
    # img_array = img_to_array(img)
    # img_array = np.expand_dims(np.array(img)[:, :, :3], axis=0)
    # img_array = img_array / 255
    # tensor_img = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float().to('mps')

    img, _, _ = threshold(img)

    # Convert predictions to a numpy array and resize to original image size
    # predictions = np.squeeze(predictions, axis=0)
    # predictions = np.argmax(predictions, axis=-1)
    # predictions = Image.fromarray(np.uint8(predictions * 255))
    # predictions = predictions.resize((img.width, img.height))

    # Save the predicted image
    img.save('predicted_image.jpg')

