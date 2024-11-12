from multibox import boxes
from matplotlib import pyplot as plt
import torch
import json
import numpy as np
import cv2

import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import no_grad
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader



impath = "intersection2.jpg"
info,image = boxes(impath)

imlist = []

for tupl in info:
    # ((x,y),w,h)
    x = tupl[0][0]
    y = tupl[0][1]
    w = tupl[1]
    h = tupl[2]
    imlist.append(image[y:y+h,x:x+w])

plt.imshow(imlist[0])
plt.show()
# plt.imshow(imlist[10])
# plt.show()
# plt.imshow(imlist[11])
# plt.show()
# plt.imshow(imlist[12])
# plt.show()
# plt.imshow(imlist[13])
# plt.show()
# plt.imshow(imlist[14])


testimg = imlist[0]

testimg = Image.fromarray(testimg)

# testimg = Image.fromarray(cv2.imread("newcar.jpg"))


#####################################
# Load a pretrained model
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
    idx_to_labels = {int(key): value[1] for key, value in class_idx.items()}
# with open("imagenet_class_index.json") as f:
#     class_idx = json.load(f)


model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # Set model to evaluation mode



# Preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess image
# input_image = Image.open(impath)
input_tensor = preprocess(testimg)
input_batch = input_tensor.unsqueeze(0)  # Create mini-batch as expected by the model

# Inference
with torch.no_grad():
    output = model(input_batch)

# Decode output
_, predicted_idx = torch.max(output, 1)
class_name = idx_to_labels[predicted_idx.item()]

car_labels = {"minivan", "pickup", "police_van", "sports_car", "convertible",
              "cab", "racer", "recreational_vehicle", "moving_van", "tow_truck", "jeep", "beach_wagon"}

if any(car_label in class_name.lower() for car_label in car_labels):
    print("The image contains a car.")
else:
    print("The image does not contain a car.")

# print("Predicted:", class_name)
#############################################

