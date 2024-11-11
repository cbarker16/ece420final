import cv2
import numpy as np

# Load the uploaded image
image_path = 'busyintersection.PNG'
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image_gray, (9,9), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, threshold1=200, threshold2=150)

# Dilate the edges to close gaps between edge segments
dilated_edges = cv2.dilate(edges, None, iterations=2)

# Apply morphological closing to make car shapes more connected and defined
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)

# Find contours in the closed edge-detected image
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define minimum contour area and aspect ratio to filter out irrelevant contours
min_contour_area = 2000  # Adjusted to capture typical car sizes
min_aspect_ratio = 0.8   # Allow for more width variation in car shapes
max_aspect_ratio = 4.0   # Limit to avoid long thin contours

# Loop over the contours and draw bounding boxes for relevant ones
for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_contour_area:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Filter based on aspect ratio
        if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            # Draw a bounding rectangle around detected cars
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save the output image with bounding boxes
output_path = 'busyintersection_with_bounding_boxes.PNG'
cv2.imwrite(output_path, image)

