import cv2
import numpy as np


def blur_background(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be loaded.")

    # Convert to grayscale for easier thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the entire image (this will serve as the background layer)
    blurred_background = cv2.GaussianBlur(image, (55, 55), 0)

    # Apply a binary threshold to create a rough mask for the foreground
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)

    # Find contours to isolate the largest contour as the main foreground object
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the foreground object
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the foreground object
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Expand the foreground mask to ensure it covers all parts of the foreground
        expanded_foreground_mask = cv2.dilate(mask, kernel, iterations=10)

        # Create an inverted mask for the background
        background_mask = cv2.bitwise_not(expanded_foreground_mask)

        # Apply the blurred background using the background mask
        background = cv2.bitwise_and(blurred_background, blurred_background, mask=background_mask)

        # Apply the sharp foreground using the expanded foreground mask
        foreground = cv2.bitwise_and(image, image, mask=expanded_foreground_mask)

        # Combine the sharp foreground with the blurred background
        result = cv2.add(foreground, background)

        # Save the final image and return it
        cv2.imwrite(output_path, result)
        print("Blurred background image saved as", output_path)
        return result
    else:
        raise ValueError("No distinct foreground object detected.")

output_image = blur_background("carimg.jpg", "output.jpg")
