from opencvtest import get_bounding_boxes
from opencvtest import get_main_bounding_box
from boundbox import *

image_path = "testimg.PNG"
# bounding_boxes = get_bounding_boxes(image_path)
#
# for idx, box in enumerate(bounding_boxes):
#     print(f"Object {idx + 1} bounding box: {box}")

bounding_box = main_bounding_box(image_path)
print("Main bounding box:", bounding_box)
display_bounding_box(image_path,bounding_box)