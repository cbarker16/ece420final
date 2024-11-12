from multibox import boxes
from matplotlib import pyplot as plt
import json
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# GOOD
# rbout.jpg
# intersection.jpg
# lastoneplz.jpg
# rbout2.jpg
# bdoneplz.jpeg

# DECENT
# helpme.jpg

# ASS
#busyintersection
# intersection2
# cluttered

# impath = "crashout.jpg"
# impath = "intersection.jpg"
impath = "bdoneplz.jpeg"
info,image = boxes(impath)

imlist = []

for tupl in info:
    # ((x,y),w,h)
    x = tupl[0][0]
    y = tupl[0][1]
    w = tupl[1]
    h = tupl[2]
    imlist.append(image[y:y+h,x:x+w])

# plt.imshow(image)
# plt.show()
# plt.imshow(imlist[1])
# plt.show()
# plt.imshow(imlist[3])
# plt.show()
# plt.imshow(imlist[4])
# plt.show()
# plt.imshow(imlist[5])
# plt.show()
# plt.imshow(imlist[6])
# plt.show()
# plt.imshow(imlist[7])
# plt.show()
# plt.imshow(imlist[10])
# plt.show()
# plt.imshow(imlist[11])
# plt.show()
# plt.imshow(imlist[12])
# plt.show()
# plt.imshow(imlist[13])
# plt.show()
# plt.imshow(imlist[14])

# cv2.imwrite("testout.PNG", imlist[3])

# testimg = imlist[0]
#
# testimg = Image.fromarray(testimg)


# testimg = Image.fromarray(cv2.imread("intersection.jpg"))
# testimg = Image.fromarray(cv2.imread("testout.PNG"))

#####################################
# Load a pretrained model
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
    idx_to_labels = {int(key): value[1] for key, value in class_idx.items()}
# with open("imagenet_class_index.json") as f:
#     class_idx = json.load(f)


# model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
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
#SINGLE IN
# input_tensor = preprocess(testimg)
# input_batch = input_tensor.unsqueeze(0)  # Create mini-batch as expected by the model

#BATCH IN
testimgs = []
for i in imlist:
    testimgs.append(preprocess(Image.fromarray(i)))

input_batch = torch.stack(testimgs)

# Inference
with torch.no_grad():
    output = model(input_batch)

# Decode output
#SINGLE OUT
# _, predicted_idx = torch.max(output, 1)
# class_name = idx_to_labels[predicted_idx.item()]
#
car_labels = {"minivan", "pickup", "police_van", "sports_car", "convertible",
              "cab", "racer", "recreational_vehicle", "moving_van", "tow_truck", "jeep", "beach_wagon"}
#
# if any(car_label in class_name.lower() for car_label in car_labels):
#     print("The image contains a car.")
# else:
#     print("The image does not contain a car.")


#BATCH OUT
labels = []
carcount = 0
_, predicted_idxs = torch.max(output, 1)  # Get the predicted index for each image in the batch
for i, predicted_idx in enumerate(predicted_idxs):
    class_name = idx_to_labels[predicted_idx.item()]  # Convert index to class name

    # Check if the predicted class name contains any of the car-related labels
    if any(car_label in class_name.lower() for car_label in car_labels):
        print(f"Image {i + 1} contains a car")
        labels.append("Car")
        carcount+=1
    else:
        print(f"Image {i + 1} does not contain a car.")
        labels.append("Not a Car")
#
# probabilities = F.softmax(output, dim=1)
#
# # Decode output
# labels = []
# _, predicted_idxs = torch.max(output, 1)  # Get the predicted index for each image in the batch
# for i, (predicted_idx, probs) in enumerate(zip(predicted_idxs, probabilities)):
#     class_name = idx_to_labels[predicted_idx.item()]  # Convert index to class name
#     confidence = probs[predicted_idx].item()  # Confidence of the top prediction
#
#     # Check if the class name is related to cars and meets the confidence threshold
#     if any(car_label in class_name.lower() for car_label in car_labels) and confidence >= 0.3:
#         print(f"Image {i + 1} contains a car with confidence {confidence:.2f}")
#         labels.append("Car")
#     else:
#         print(f"Image {i + 1} does not contain a car (confidence {confidence:.2f}).")
#         labels.append("Not a Car")
#


#############################################

for idx, tupl in enumerate(info):
    x = tupl[0][0]
    y = tupl[0][1]
    w = tupl[1]
    h = tupl[2]

    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Put the label in the bounding box
    cv2.putText(image, labels[idx], (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# cv2.putText(image, "CAR COUNT: " + str(carcount), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 01.25, (255, 0, 0), 2)

print("Car count = " + str(carcount))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()


#