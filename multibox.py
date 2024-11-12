import cv2
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt


def compute_iou(box1, box2):
    xy1, w1, h1 = box1  # x = starting horizontal position of box
    xy2, w2, h2 = box2  # y = starting vertical position of bos
    x1, y1 = xy1  # w = width of box
    x2, y2 = xy2  # h = height of box

    intersect_x1 = max(x1, x2)  # find coordinates of intersection rectangle
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1 + w1, x2 + w2)
    intersect_y2 = min(y1 + h1, y2 + h2)

    intersect_w = max(0, intersect_x2 - intersect_x1)  # find area of intersection rectangle
    intersect_h = max(0, intersect_y2 - intersect_y1)
    intersection_area = intersect_w * intersect_h

    area1 = w1 * h1  # areas of bounding boxes
    area2 = w2 * h2

    union_area = area1 + area2 - intersection_area  # union area

    if union_area == 0:  # if rectangles don't intersect, IoU = 0
        return 0
    return intersection_area / union_area


# Apply Non-Maximum Suppression
def non_max_suppression(bounding_boxes, scores, iou_threshold):
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i],
                            reverse=True)  # sort boxes in descending order based on score
    selected_boxes = []

    while sorted_indices:
        current = sorted_indices.pop(0)
        selected_boxes.append(bounding_boxes[current])

        remaining_boxes = []
        for i in sorted_indices:
            iou = compute_iou(bounding_boxes[current], bounding_boxes[i])
            # print(iou)
            if iou < iou_threshold:
                remaining_boxes.append(i)
            # print(remaining_boxes)
        sorted_indices = remaining_boxes

    return selected_boxes


def boxes(path):
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_gray, (9, 9), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, threshold1=175, threshold2=150)

    # Dilate the edges to close gaps between edge segments
    dilated_edges = cv2.dilate(edges, None, iterations=2)

    # Apply morphological closing to make car shapes more connected and defined
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed edge-detected image
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define minimum contour area and aspect ratio to filter out irrelevant contours
    min_contour_area = 2000  # Adjusted to capture typical car sizes
    min_aspect_ratio = 0.62  # Allow for more width variation in car shapes
    max_aspect_ratio = 4.0  # Limit to avoid long thin contours

    # Loop over the contours and draw bounding boxes for relevant ones
    boundingboxinfo = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Filter based on aspect ratio
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                # Draw a bounding rectangle around detected cars
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                boundingboxinfo.append(((x, y), w, h))
        output_path = 'busyintersection_with_bounding_boxes.PNG'
        cv2.imwrite(output_path, image)

    # IoU and NMS

    integral_image = cv2.integral(edges)  # create integral image

    def compute_score(x, y, w, h, integral_img):  # function to compute bounding box score
        sum_edges = integral_img[y + h, x + w] - integral_img[y, x + w] - integral_img[y + h, x] + integral_img[y, x]
        perimeter = 2 * (w + h)
        kappa = 1.5  # Bias term for larger boxes
        return sum_edges / (perimeter ** kappa)

    scores = []
    # print(boundingboxinfo)
    for (xy, w, h) in boundingboxinfo:
        x, y = xy
        score = compute_score(x, y, w, h, integral_image)
        scores.append(score)

    iou_threshold = 0.5
    final_boxes = non_max_suppression(boundingboxinfo, scores, iou_threshold)

    return final_boxes, image


# Load the uploaded image
# image_path = 'rbout.jpg'
#
# info, newimage = boxes(image_path)
# # pprint(info)
# plt.imshow(newimage)
# plt.show()