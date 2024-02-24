#!/usr/bin/env/ python3
from typing import List,Dict,Tuple
import numpy as np
from utils import crop_image_to_aspect_ratio, transform_json_to_dict
from sift import SIFTImageComparator
from craft import CRAFTAutoCrop

"""
Assumptions for input parameters:
1. Crop Coordinates list: the top left coordinate is the first in the list
    then, the bottom right one. This is a List of type Tuple. [(x1,y1),(x2,y2),...]
2. Aspect Rato: this is a float that represents width/height, not tuple
"""

def CLASSIFY_BADGE_PYTHON(crop_coordinates:List[float], 
                          aspect_ratio:float, 
                          image_path:str, 
                          json_path:str='../environmental_labels_data.json',
                          craft_output_dir:str='data_craft/') -> str:
    
    SIFT = SIFTImageComparator()
    # CRAFT = CRAFTAutoCrop(output_dir=craft_output_dir)
    
    IMAGE=SIFT.read_image(image_path)

    if (crop_coordinates is not None) and (aspect_ratio is not None):

        top_left:tuple=crop_coordinates[0]
        bottom_right:tuple=crop_coordinates[1]
        IMAGE:np.ndarray = crop_image_to_aspect_ratio(IMAGE,
                                                    top_left,
                                                    bottom_right,
                                                    aspect_ratio)
    
    (anchor_keyp, anchor_desc) = SIFT.generate_sift_pair(IMAGE)
    REFERENCE_DESCRIPTORS:Dict[str,dict] = transform_json_to_dict(json_path)

    print(type(REFERENCE_DESCRIPTORS['Bio Quebec']['keypoints']))
    print(type(REFERENCE_DESCRIPTORS['Bio Quebec']['descriptors']))
    print(type(anchor_desc))


    # pair-wise comparison of descriptors,keypoints
    similarities:List[Tuple[str,float]]= [ (properties['type'], SIFT.compare(anchor_desc, 
                                                                             properties['descriptors'], 
                                                                             anchor_keyp, 
                                                                             properties['keypoints'])) 
                                            for _, properties in REFERENCE_DESCRIPTORS.items()]
    
    most_similar_badge = max(similarities, key=lambda x: x[1])[0]

    return most_similar_badge


if __name__ == '__main__':
    PATH = '../data/certified-vegan.png'

    res = CLASSIFY_BADGE_PYTHON(crop_coordinates=None,
                                aspect_ratio=None,
                                image_path=PATH)
    
    print(res)
    