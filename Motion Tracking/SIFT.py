import cv2
import numpy as np

# Create SIFT object and BF matcher to find features and knn matches
sift = cv2.SIFT_create()
bruteForce = cv2.BFMatcher()


def get_features(frame: np.ndarray):
    """Find features in the frame and return keypoints and descriptors"""
    keyPoints, descriptors = sift.detectAndCompute(frame, None)
    return keyPoints, descriptors


def match_features(des1, des2):
    """Match features in two images given their descriptors"""
    matches = bruteForce.knnMatch(des1, des2, k=2)
    # Apply ratio test to find good matches
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good
