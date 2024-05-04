import cv2
import torch

net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5,
                     map_location=torch.device('cpu'))

file = '/Users/sofia/PycharmProjects/smartDots/data/original_all_accordance/2021_411_94fcab82-17e1-408b-ac00-6ad5c36f550b.jpg'

img = cv2.imread(file)

img_resized = cv2.resize(img, (572, 572))

tensor_img = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float().to('mps')
# reshape(2, 1, 0).float().to('mps')
print(tensor_img.shape)
unet_img = net(tensor_img)

