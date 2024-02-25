#!/usr/bin/env/ python3
import numpy as np
from typing import Tuple
import cv2

class SIFTImageComparator:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def read_image(self, image_path: str) -> np.ndarray:
        """Reads an image from a path and converts it to grayscale."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        return image
    
    def generate_sift_pair(self, image:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        (k, d) = self.sift.detectAndCompute(image, None)
        return (k, d)

    def compare(self, descriptor_1:np.ndarray, descriptor_2:np.ndarray, keypoints1:np.ndarray, keypoints2:np.ndarray) -> float:
        threshold:float = 0.75
        neighbors:int = 2

        descriptor_1 = descriptor_1.astype(np.float32)
        descriptor_2 = descriptor_2.astype(np.float32)

        # Match descriptors using FLANN matcher
        # https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        matches = self.flann.knnMatch(descriptor_1, descriptor_2, k=neighbors)

        # Apply Lowe's ratio test to filter matches
        good_matches = [m for m, n in matches if m.distance < threshold * n.distance]

        # Calculate similarity as a ratio of good matches to total matches
        total_matches = min(len(keypoints1), len(keypoints2))
        similarity = len(good_matches) / total_matches if total_matches > 0 else 0

        return similarity

    def fit_transform(self, img_path1: str, img_path2: str) -> float:
        """Computes the similarity between two images using SIFT features."""
        # Load and convert images to grayscale
        img1 = self.read_image(img_path1)
        img2 = self.read_image(img_path2)

        # Detect and compute key points and descriptors with SIFT
        keypoints1, descriptors1 = self.generate_sift_pair(img1)
        keypoints2, descriptors2 = self.generate_sift_pair(img2)
        score:float=self.compare(descriptors1, descriptors2, keypoints1, keypoints2)

        return score
