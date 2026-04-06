import numpy as np
import cv2


class ViewTransformer():
    """
    Converts pixel coordinates from the video frame into real-world field
    coordinates (in meters).

    The camera films the pitch at an angle, so distances in pixels are not
    uniform — 1 pixel near the top of the frame represents more real-world
    distance than 1 pixel near the bottom. This class uses a perspective
    transform to correct for that, mapping a trapezoid in pixel space
    (how the field looks on camera) to a rectangle in meter space
    (how the field actually is).

    Real-world dimensions used:
        - Court width:  68 meters
        - Court length: 23.32 meters (the visible section of the pitch)
    """

    def __init__(self):
        # Real-world dimensions of the visible field section (in meters)
        court_width = 68
        court_length = 23.32

        # The 4 corners of the visible field in pixel coordinates.
        # These are manually measured from the video frame — they form a trapezoid
        # because of the camera's perspective angle.
        self.pixel_vertices = np.array([
            [110, 1035],   # bottom-left
            [265, 275],    # top-left
            [910, 260],    # top-right
            [1640, 915]    # bottom-right
        ])

        # The same 4 corners but in real-world meter coordinates.
        # This is a perfect rectangle since the actual field is flat.
        self.target_vertices = np.array([
            [0, court_width],       # bottom-left
            [0, 0],                 # top-left
            [court_length, 0],      # top-right
            [court_length, court_width]  # bottom-right
        ])

        # cv2 requires float32 for perspective transform calculations
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Compute the perspective transform matrix — the math that maps
        # pixel trapezoid → real-world rectangle
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        """
        Converts a single pixel coordinate into real-world field coordinates (meters).

        Points outside the visible field area return None — they can't be
        meaningfully mapped to real-world coordinates.

        Args:
            point: A pixel coordinate as a numpy array [x, y].

        Returns:
            Transformed point as a numpy array [x_meters, y_meters],
            or None if the point is outside the field boundary.
        """

        p = (int(point[0]), int(point[1]))

        # Check if the point falls inside our defined field polygon.
        # pointPolygonTest returns >= 0 if inside or on the boundary, < 0 if outside.
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # cv2.perspectiveTransform requires shape (N, 1, 2) — reshape accordingly
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)

        # Squeeze back from (1, 1, 2) → (1, 2) for easier downstream use
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """
        Converts every tracked object's adjusted position from pixels to real-world
        meters and stores it back into the tracks dictionary.

        Uses 'position_adjusted' (which already has camera movement removed) as input,
        so the resulting 'position_transformed' represents the object's true location
        on the field in meters — free from both camera angle and camera movement distortion.

        Args:
            tracks: Nested dict of {object: [{track_id: track_info}, ...]}
                    Each track_info must have a 'position_adjusted' key.

        Modifies tracks in-place by adding 'position_transformed' to each track entry.
        Points outside the field boundary are stored as None.
        """

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():

                    # Start from the camera-movement-corrected position
                    position = track_info['position_adjusted']
                    position = np.array(position)

                    # Convert pixel position → real-world meters
                    position_transformed = self.transform_point(position)

                    # Flatten from numpy array to a plain [x, y] list for consistency
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()

                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed