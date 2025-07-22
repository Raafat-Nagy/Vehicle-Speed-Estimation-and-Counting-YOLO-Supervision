import numpy as np
import supervision as sv


class Annotator:
    """
    A flexible video annotation utility for computer vision pipelines.

    Supports drawing:
    - Bounding boxes
    - Labels
    - Single-class and multi-class line zones
    - Polygon zones
    - Object traces (trajectories)
    """

    def __init__(
        self,
        resolution_wh: tuple[int, int],
        box_annotator: bool = True,
        label_annotator: bool = False,
        line_annotator: bool = False,
        multi_class_line_annotator: bool = False,
        trace_annotator: bool = False,
        trace_length: int = 20,
        polygon_zone=None,
    ):
        """
        Initialize the Annotator with desired annotation settings.

        Args:
            resolution_wh (tuple[int, int]): Video resolution (width, height).
            box_annotator (bool): Enable bounding box annotation.
            label_annotator (bool): Enable label annotation.
            line_annotator (bool): Enable single-class line zone annotation.
            multi_class_line_annotator (bool): Enable multi-class line zone annotation.
            trace_annotator (bool): Enable trajectory annotation for tracked objects.
            trace_length (int): Maximum number of positions to keep for traces.
            polygon_zone (list[tuple[int, int]], optional): Polygon zone coordinates.
        """
        self.thickness = sv.calculate_optimal_line_thickness(resolution_wh)
        self.text_scale = sv.calculate_optimal_text_scale(resolution_wh)

        # Initialize annotators
        self.box_annotator = (
            sv.BoxAnnotator(thickness=self.thickness) if box_annotator else None
        )
        self.label_annotator = (
            sv.LabelAnnotator(
                text_thickness=self.thickness,
                text_scale=self.text_scale,
                text_position=sv.Position.BOTTOM_CENTER,
            )
            if label_annotator
            else None
        )
        self.line_annotator = (
            sv.LineZoneAnnotator(
                thickness=self.thickness,
                text_thickness=self.thickness,
                text_scale=self.text_scale,
            )
            if line_annotator
            else None
        )
        self.multi_class_line_annotator = (
            sv.LineZoneAnnotatorMulticlass(
                text_thickness=self.thickness,
                text_scale=self.text_scale,
            )
            if multi_class_line_annotator
            else None
        )
        self.trace_annotator = (
            sv.TraceAnnotator(thickness=self.thickness, trace_length=trace_length)
            if trace_annotator
            else None
        )
        self.polygon_annotator = (
            sv.PolygonZoneAnnotator(
                zone=sv.PolygonZone(polygon_zone),
                thickness=self.thickness,
                color=sv.Color.GREEN,
                display_in_zone_count=False,
            )
            if polygon_zone is not None
            else None
        )

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: list[str] | None = None,
        line_zones: list[sv.LineZone] | None = None,
        multi_class_zones: list[sv.LineZone] | None = None,
    ) -> np.ndarray:
        """
        Annotate a video frame with selected features.

        Args:
            frame (np.ndarray): The video frame to annotate.
            detections (sv.Detections): Detection objects for annotation.
            labels (list[str], optional): Labels for objects.
            line_zones (list[sv.LineZone], optional): Single-class line zones.
            multi_class_zones (list[sv.LineZone], optional): Multi-class line zones.

        Returns:
            np.ndarray: Annotated video frame.
        """
        annotated_frame = frame.copy()

        # Draw bounding boxes
        if self.box_annotator:
            annotated_frame = self.box_annotator.annotate(annotated_frame, detections)

        # Draw labels
        if self.label_annotator:
            annotated_frame = self.label_annotator.annotate(
                annotated_frame, detections, labels=labels
            )

        # Draw traces (trajectories)
        if self.trace_annotator:
            annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)

        # Draw line zones (single-class)
        if self.line_annotator and line_zones:
            for zone in line_zones:
                annotated_frame = self.line_annotator.annotate(
                    annotated_frame, line_counter=zone
                )

        # Draw multi-class line zones
        if self.multi_class_line_annotator and multi_class_zones:
            annotated_frame = self.multi_class_line_annotator.annotate(
                annotated_frame, line_zones=multi_class_zones
            )

        # Draw polygon zone
        if self.polygon_annotator:
            annotated_frame = self.polygon_annotator.annotate(annotated_frame)

        return annotated_frame
