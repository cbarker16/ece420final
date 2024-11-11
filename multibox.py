import cv2
import numpy as np

# Load the image
image_path = 'carimg.jpg'
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Dilate the edges to close gaps between edge segments
dilated_edges = cv2.dilate(edges, None, iterations=2)

# Find contours in the dilated edge-detected image
contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define minimum contour area and aspect ratio to filter out irrelevant contours
min_contour_area = 10000
min_aspect_ratio = 1.5  # Width should be larger than height for a car

# Loop over the contours and draw a bounding rectangle for large ones
for contour in contours:
    # Filter contours by area
    if cv2.contourArea(contour) > min_contour_area:
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Filter based on aspect ratio
        if aspect_ratio > min_aspect_ratio:
            # Draw a bounding rectangle around the car
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Save the output image
output_path = 'newcar_with_bounding_boxes.jpg'
cv2.imwrite(output_path, image)
