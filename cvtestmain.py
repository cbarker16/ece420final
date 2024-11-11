from opencvtest import get_bounding_boxes
from opencvtest import get_main_bounding_box
from boundbox import *
# from multibox import *

image_path = "testimg.PNG"
car = "newcar.jpg"
# bounding_boxes = get_bounding_boxes(image_path)
#
# for idx, box in enumerate(bounding_boxes):
#     print(f"Object {idx + 1} bounding box: {box}")

bounding_box = main_bounding_box(car)
print("Main bounding box:", bounding_box)
display_bounding_box(car,bounding_box)

# bounding_boxes = main_bounding_boxes(car)
# display_bounding_boxes(car, bounding_boxes)
