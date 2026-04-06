def get_center_of_bbox(bbox):
    """
    Returns the center point of a bounding box.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        (x, y): Center coordinates as integers.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    """
    Returns the width of a bounding box.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        width: Horizontal size of the box in pixels.
    """
    return bbox[2] - bbox[0]


def measure_distance(p1, p2):
    """
    Calculates the straight-line (Euclidean) distance between two points.

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        distance: Euclidean distance as a float.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2):
    """
    Returns the separate horizontal and vertical distance between two points.

    Unlike measure_distance() which returns a single value, this returns
    the x and y components separately — useful for camera movement where
    we need to know the direction (left/right, up/down).

    Args:
        p1: First point (x, y)
        p2: Second point (x, y)

    Returns:
        (dx, dy): Horizontal and vertical distance from p1 to p2.
    """
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox):
    """
    Returns the bottom-center point of a bounding box — approximating foot position.

    Used instead of the full center for players since they stand on the ground.
    Bottom-center is more accurate for field distance calculations.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        (x, y): Bottom-center coordinates as integers.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)