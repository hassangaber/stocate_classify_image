import numpy as np
from typing import List, Tuple
import json
import math
import cv2

def get_interesting_coordinates(coords: List[float], threshold:float=0.5) -> List[Tuple[float, float]]:
    """
    Find the top left and bottom right coordinates from a list of coordinates
    based on an image coordinate system with the origin at the top left corner.
    """
    coordinate_pairs = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

    # Initialize with the first pair to have something to compare with
    top_left = coordinate_pairs[0]
    bottom_right = coordinate_pairs[0]

    for x, y in coordinate_pairs[1:]:
        # For top left, look for the smallest x and then the smallest y
        if x < top_left[0] or (x == top_left[0] and y < top_left[1]):
            top_left = (x, y)
        
        # For bottom right, look for the largest x and then the largest y
        if x > bottom_right[0] or (x == bottom_right[0] and y > bottom_right[1]):
            bottom_right = (x, y)
    
    distance = math.sqrt((bottom_right[0] - top_left[0])**2 + (bottom_right[1] - top_left[1])**2)
    
    # Threshold check
    if distance < threshold:
        raise ValueError("The selected points are too close to each other based on the given threshold.")

    return [top_left, bottom_right]


def crop_image_to_aspect_ratio(image:np.ndarray, top_left:tuple, bottom_right:tuple, aspect_ratio:float)->np.ndarray:
    """
    Crop an image to a specified aspect ratio.

    :param image: np.ndarray, the image to crop.
    :param top_left: tuple of (x, y), the top-left coordinate of the crop area.
    :param bottom_right: tuple of (x, y), the bottom-right coordinate of the crop area.
    :param aspect_ratio: float, the desired aspect ratio (width / height).
    :return: np.ndarray, the cropped image.
    """
    # Calculate the initial crop width and height
    initial_width = bottom_right[0] - top_left[0]
    initial_height = bottom_right[1] - top_left[1]
    
    # Calculate the current aspect ratio
    current_aspect_ratio = initial_width / initial_height

    print(current_aspect_ratio)
    
    # Adjust the crop area to match the desired aspect ratio
    if current_aspect_ratio > aspect_ratio:
        # Current crop is too wide
        new_width = int(aspect_ratio * initial_height)
        width_reduction = initial_width - new_width
        top_left = (top_left[0] + width_reduction // 2, top_left[1])
        bottom_right = (bottom_right[0] - width_reduction // 2, bottom_right[1])
        
    elif current_aspect_ratio < aspect_ratio:
        # Current crop is too tall
        new_height = int(initial_width / aspect_ratio)
        height_reduction = initial_height - new_height
        top_left = (top_left[0], top_left[1] + height_reduction // 2)
        bottom_right = (bottom_right[0], bottom_right[1] - height_reduction // 2)
    

    cropped_image = image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]
    
    return cropped_image

def transform_json_to_dict(json_file_path:str)->dict:
    """
    Transforms a JSON file into a Python dictionary with a specific structure.
    
    :param json_file_path: str, path to the JSON file.
    :return: dict, transformed dictionary.
    """
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Initialize the result dictionary
    transformed_dict = {}
    
    # Iterate through the JSON data and restructure it
    for badge_name, properties in data.items():
        badge_property = properties.get('badges')
        if badge_property:  # Check if badge_property exists
            transformed_dict[badge_name] = {
                "type": badge_property,
                "keypoints": np.array(properties.get('keypoints', [])),
                "descriptors": np.array(properties.get('descriptors', []))
            }
    
    return transformed_dict

def resize_image(image: np.ndarray, coords:List[float], scale: float = 0.95) -> Tuple[np.ndarray,List[float]]:
    """
    Resize the image by a percentage based on the float value (0.1 -> 10%).
    
    Parameters:
        image (np.ndarray): The input image to resize.
        scale (float): The scaling factor for resizing the image.
    
    Returns:
        np.ndarray: The resized image.
    """
    # Calculate the new dimensions
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    new_dimensions = (width, height)
    
    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    print(f'Image resized from {image.shape} to {resized_image.shape}')

    coords = [c*scale for c in coords]

    return (resized_image, coords)



if __name__ == "__main__":
    l = [4.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 2.0]
    # [1.0, 3.0, 4.0, 2.0, 2.5, 5.0, 3.5, 1.0]
    print(get_interesting_coordinates(l, 2.0))

