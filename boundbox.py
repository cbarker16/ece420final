import numpy as np
from PIL import Image
import cv2

def load_image_grayscale(image_path):
    """
    Load an image and convert it to grayscale manually.
    """
    img = Image.open(image_path)
    img = img.convert("L")  # Convert to grayscale
    img_data = np.array(img)
    return cv2.equalizeHist(img_data)
    # return img_data

def blur(image):
    # blurfilter = np.array([[1/16, 1/8, 1/16],
    #                     [1/8, 1/4, 1/8],
    #                     [1/16, 1/8, 1/16]])
    blurfilter = (1/100)* np.ones((10,10))

    padded_img = np.pad(image, 1, mode='constant', constant_values=0)
    img_fft = fft2(padded_img)
    blurfft = fft2(blurfilter, s=padded_img.shape)
    blurred = ifft2(img_fft*blurfft)
    return blurred.astype(int)
    # return image

def apply_threshold(img_data, threshold=127):
    """
    Apply a binary threshold to a grayscale image.
    Pixels above the threshold become 1 (white), and below become 0 (black).
    """

    binary_img = np.where(img_data > threshold, 1, 0)
    return binary_img


# def detect_edges_sobel(binary_img):
#     """
#     Simple edge detection using Sobel-like operators.
#     """
#     sobel_x = np.array([[-1, 0, 1],
#                         [-2, 0, 2],
#                         [-1, 0, 1]])
#     sobel_y = np.array([[-1, -2, -1],
#                         [0, 0, 0],
#                         [1, 2, 1]])
#
#     # Pad the image to apply the filter on the edges
#     padded_img = np.pad(binary_img, 1, mode='constant', constant_values=0)
#     edges = np.zeros_like(binary_img)
#
#     # Convolve each pixel (excluding padded borders)
#     for i in range(1, padded_img.shape[0] - 1):
#         for j in range(1, padded_img.shape[1] - 1):
#             # Get the 3x3 region
#             region = padded_img[i - 1:i + 2, j - 1:j + 2]
#             # Apply Sobel operators
#             gx = np.sum(region * sobel_x)
#             gy = np.sum(region * sobel_y)
#             # Calculate edge intensity
#             edges[i - 1, j - 1] = np.sqrt(gx ** 2 + gy ** 2)
#
#     # Normalize edges to binary for contour detection
#     edge_threshold = 0.1 * edges.max()
#     edges_binary = (edges > edge_threshold).astype(int)
#     return edges_binary

import numpy as np
from scipy.fft import fft2, ifft2
from matplotlib import pyplot as plt
import time


def detect_edges_sobel(binary_img):
    """
    Faster edge detection using FFT-based convolution with Sobel operators.
    """
    # Define Sobel filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Pad the binary image to avoid edge effects
    padded_img = np.pad(binary_img, 1, mode='constant', constant_values=0)

    # Compute the 2D FFT of the padded image and Sobel kernels
    img_fft = fft2(padded_img)
    sobel_x_fft = fft2(sobel_x, s=padded_img.shape)
    sobel_y_fft = fft2(sobel_y, s=padded_img.shape)

    # Perform element-wise multiplication in the frequency domain
    gx = ifft2(img_fft * sobel_x_fft).real
    gy = ifft2(img_fft * sobel_y_fft).real

    # Calculate the gradient magnitude
    edges = np.sqrt(gx ** 2 + gy ** 2)
    # plt.figure()
    # plt.imshow(edges,cmap='gray')
    # time.sleep(10)

    # Threshold edges to binary for contour detection
    edge_threshold = 0.5 * edges.max()
    edges_binary = (edges > edge_threshold).astype(int)

    return edges_binary


def find_bounding_box(edges_binary):
    """
    Find the bounding box around the detected edges.
    """
    # Get coordinates of edge pixels
    y_coords, x_coords = np.where(edges_binary == 1)

    # If no edges detected, return None
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None

    # Find the bounding box coordinates
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    return (x_min, y_min, x_max - x_min, y_max - y_min)

def find_bounding_boxes(edges_binary):
    """
    Find the bounding box around the detected edges.
    """
    # Get coordinates of edge pixels
    y_coords, x_coords = np.where(edges_binary == 1)

    # If no edges detected, return None
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None

    # Find the bounding box coordinates
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def main_bounding_box(image_path):
    """
    Detects the main bounding box of the subject in an image from scratch.
    """
    # Load and preprocess image
    img_data = load_image_grayscale(image_path)
    blurred_data = blur(img_data)
    plt.figure()
    plt.imshow(blurred_data, cmap='gray')
    plt.title("Blurred Image")
    plt.show()
    binary_img = apply_threshold(blurred_data)
    # binary_img = apply_threshold(img_data)
    edges_binary = detect_edges_sobel(binary_img)

    plt.figure()
    plt.imshow(edges_binary, cmap='gray')
    plt.title("Binary Edge Image")
    plt.show()

    # Find and return bounding box
    bounding_box = find_bounding_box(edges_binary)
    return bounding_box


from PIL import Image, ImageDraw
import numpy as np

def display_bounding_box(image_path, bounding_box):
    """
    Displays an image with a bounding box drawn on it.

    Parameters:
    - image_path: Path to the image file.
    - bounding_box: A tuple (x, y, width, height) representing the bounding box.
    """
    # Load the image
    img = Image.open(image_path)

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Unpack the bounding box coordinates
    x, y, width, height = bounding_box

    # Draw the bounding box as a rectangle
    draw.rectangle([x, y, x + width, y + height], outline="red", width=3)

    # Display the image with the bounding box
    img.show()

