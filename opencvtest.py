import cv2
import numpy as np

def get_bounding_boxes(image_path):
    """
    Detects objects in an image and returns their bounding boxes.

    Parameters:
    - image_path: Path to the input image file

    Returns:
    - bounding_boxes: List of bounding boxes as (x, y, width, height)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be loaded")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to hold bounding boxes
    bounding_boxes = []

    # Loop over contours and extract bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if(w*h>=200):
            bounding_boxes.append((x, y, w, h))

        # Optionally, draw the bounding box on the image for visualization
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result with bounding boxes
    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return bounding_boxes


import cv2
import numpy as np

import cv2
import numpy as np

def get_main_bounding_box(image_path):
    """
    Detects the main subject in an image and returns a bounding box
    that encompasses the main detected contours.

    Parameters:
    - image_path: Path to the input image file

    Returns:
    - main_bounding_box: A tuple representing the bounding box as (x, y, width, height)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be loaded")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours by setting a minimum area threshold
    min_contour_area = 500  # Adjust this threshold based on the size of the main object
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # If no contours remain after filtering, return an empty result
    if not filtered_contours:
        return None

    # Initialize the coordinates for the combined bounding box
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    # Combine all filtered contours into one bounding box
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Calculate width and height of the main bounding box
    width = x_max - x_min
    height = y_max - y_min

    # Draw the main bounding box on the image for visualization
    main_bounding_box = (x_min, y_min, width, height)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the result with the main bounding box
    cv2.imshow("Main Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return main_bounding_box
