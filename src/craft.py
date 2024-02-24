#!/usr/bin/env/ python3
from craft_text_detector import Craft

class CRAFTAutoCrop:
    def __init__(self, output_dir:str = 'data_craft/'):
        self.craft = Craft(output_dir=output_dir, 
                           crop_type="poly", 
                           cuda=False,
                           rectify=True)
    
    def predict(self, image_path:str) -> str:
        # apply craft text detection and export detected regions to output directory
        prediction_result = self.craft.detect_text(image_path)

        # unload models from ram/gpu
        self.craft.unload_craftnet_model()
        self.craft.unload_refinenet_model()
        return prediction_result
