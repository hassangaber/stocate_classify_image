import numpy as np
import json

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
    
    # Crop the image
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
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