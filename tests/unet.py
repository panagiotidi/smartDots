# import cv2
# import torch
#
# from classifier.models.model_ResNetUNet import ResNetUNet
#
# from tensorflow._api.v2 import image
from tensorflow._api.v2 import image

from main_classifier.models.model_unet import unet_model

file = '/Users/sofia/PycharmProjects/smartDots/data/original_all_accordance/2018_195_2c99947d-a34e-4c8e-b0f6-b52bdfa06b5a.jpg'

# img = cv2.imread(file)
#
# model = ResNetUNet().to('mps')
#
# img_resized = cv2.resize(img, (512, 512))
#
# # img.resize((572, 572))
# # print(img_resized)
# tensor_img = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float().to('mps')
# # reshape(2, 1, 0).float().to('mps')
# print(tensor_img.shape)
# unet_img = model(tensor_img)
# print(unet_img.shape)
#
# cv2.imshow('Binary image', unet_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()#
#
import numpy as np
from PIL import Image
# from tensorflow.keras.preprocessing import image

# Load the image
img = Image.open(file)
# Preprocess the image
img = img.resize((572, 572))
# img_array = img_to_array(img)
img_array = np.expand_dims(np.array(img)[:, :, :3], axis=0)
img_array = img_array / 255.

# Load the model
model = unet_model(input_shape=(572, 572, 3), num_classes=2)

# Make predictions
predictions = model.predict(img_array)
print(predictions.shape)

# Convert predictions to a numpy array and resize to original image size
predictions = np.squeeze(predictions, axis=0)
predictions = np.argmax(predictions, axis=-1)
predictions = Image.fromarray(np.uint8(predictions * 255))
predictions = predictions.resize((img.width, img.height))

# Save the predicted image
predictions.save('predicted_image.jpg')
predictions