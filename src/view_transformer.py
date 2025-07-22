import cv2
import numpy as np


class ViewTransformer:
    """
    Handles perspective transformation of points from source view to target view.
    Useful for converting image coordinates to a top-down view for speed estimation.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Initialize the transformation matrix from source to target points.

        Args:
            source (np.ndarray): Array of 4 points defining the original perspective.
            target (np.ndarray): Array of 4 points defining the target perspective.
        """
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.source, self.target
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to a set of points.

        Args:
            points (np.ndarray): Array of shape (N, 2) containing points in source view.

        Returns:
            np.ndarray: Transformed points in target view, shape (N, 2).
        """
        if points.size == 0:
            return points

        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
            points, self.transformation_matrix
        )

        return transformed_points.reshape(-1, 2)
