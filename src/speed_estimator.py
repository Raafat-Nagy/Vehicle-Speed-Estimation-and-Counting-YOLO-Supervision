import numpy as np
import supervision as sv
from collections import defaultdict, deque
from .view_transformer import ViewTransformer


class SpeedEstimator:
    """
    Estimates object speed based on movement across frames using perspective transformation.
    """

    def __init__(
        self,
        fps: int,
        view_transformer: ViewTransformer,
        max_history_seconds: int = 1,
    ):
        """
        Args:
            fps (int): Video frames per second.
            view_transformer (ViewTransformer): Instance for perspective transformation.
            max_history_seconds (int): Max time window to calculate speed (in seconds).
        """
        self.fps = fps
        self.view_transformer = view_transformer
        self.coordinates = defaultdict(
            lambda: deque(maxlen=int(fps * max_history_seconds))
        )

    def calculate_speed(self, tracker_id: int) -> float | None:
        """
        Calculate speed for a specific tracker ID.

        Returns:
            float | None: Speed in km/h if enough data, else None.
        """
        coords = self.coordinates[tracker_id]
        if len(coords) > self.fps / 2:  # Ensure enough movement history
            start, end = coords[0], coords[-1]
            # Euclidean distance in transformed space
            distance = np.linalg.norm(end - start)  # in meters

            time = len(coords) / self.fps  # (N/N) * S = S
            speed = (distance / time) * 3.6  # Convert m/s to km/h
            return int(speed)
        return None

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update object positions and compute speed for current frame.

        Args:
            detections (sv.Detections): Current frame detections.

        Returns:
            sv.Detections: Updated detections with 'speed' in data dictionary.
        """
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = self.view_transformer.transform_points(points)

        speeds_for_frame = []
        for tracker_id, point in zip(detections.tracker_id, points):
            self.coordinates[tracker_id].append(point)
            speed = self.calculate_speed(tracker_id)
            speeds_for_frame.append(speed if speed else 0)

        detections.data["speed"] = np.array(speeds_for_frame)
        return detections
