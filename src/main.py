#!/usr/bin/env/ python3
from typing import List, Dict, Tuple, Union
import numpy as np
from utils import crop_image_to_aspect_ratio, transform_json_to_dict, get_interesting_coordinates
from sift import SIFTImageComparator
# from craft import CRAFTAutoCrop


def CLASSIFY_BADGE_PYTHON(crop_coordinates:List[float], 
                          aspect_ratio:List[float], 
                          image_path:str, 
                          json_path:str='../environmental_labels_data.json',
                          craft_output_dir:str='data_craft/') -> Union[str, None]:
    try:
        SIFT = SIFTImageComparator()
        # CRAFT = CRAFTAutoCrop(output_dir=craft_output_dir)
        
        IMAGE=SIFT.read_image(image_path)

        # if the image has cropped ROI's, crop the image and returned the cropped version
        if (crop_coordinates is not None) and (aspect_ratio is not None):
            print(f'Initial image shape: {IMAGE.shape}')

            AR = aspect_ratio[0] / aspect_ratio[1]
            crop_coordinates = get_interesting_coordinates(crop_coordinates)
            top_left:tuple=crop_coordinates[0]
            bottom_right:tuple=crop_coordinates[1]
            IMAGE:np.ndarray = crop_image_to_aspect_ratio(IMAGE,
                                                        top_left,
                                                        bottom_right,
                                                        AR)
            
            print(f'Cropped image shape: {IMAGE.shape}')
        
        # SIFT dictionaries to compare images
        (anchor_keyp, anchor_desc) = SIFT.generate_sift_pair(IMAGE)
        REFERENCE_DESCRIPTORS:Dict[str,dict] = transform_json_to_dict(json_path)

        # pair-wise comparison of descriptors,keypoints
        similarities:List[Tuple[str,float]]= [ (properties['type'], SIFT.compare(anchor_desc, 
                                                                                properties['descriptors'], 
                                                                                anchor_keyp, 
                                                                                properties['keypoints'])) 
                                                for _, properties in REFERENCE_DESCRIPTORS.items()]
        
        # find the most similar badge
        most_similar_badge = max(similarities, key=lambda x: x[1])[0]

        return most_similar_badge
    
    except:

        print('Python: CLASSIFY_BADGE_PYTHON returned None due to error encountered.')
        return None
             


if __name__ == '__main__':
    PATH = '../data/certified-vegan.png'

    res = CLASSIFY_BADGE_PYTHON(crop_coordinates=None,
                                aspect_ratio=None,
                                image_path=PATH)
    
    print(res)
    