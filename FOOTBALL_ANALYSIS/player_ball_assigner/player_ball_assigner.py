import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner():
    """
    Determines which player currently has possession of the ball.

    Uses bounding box distances to find the closest player to the ball.
    If no player is within the allowed range, no assignment is made.
    """

    def __init__(self):
        # Maximum distance (in pixels) a player can be from the ball to be considered
        # "in possession". Players further than this are ignored entirely.
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        """
        Finds which player is closest to the ball and assigns possession to them.

        Instead of measuring from the player's center, we measure from the
        bottom-left and bottom-right corners of their bounding box (feet level).
        This is more accurate since the ball rolls on the ground near the feet,
        not at the player's vertical center.

        Args:
            players:    Dict of {player_id: {'bbox': [x1, y1, x2, y2], ...}}
            ball_bbox:  Bounding box of the ball [x1, y1, x2, y2]

        Returns:
            assigned_player:  The player_id of the player closest to the ball,
                              or -1 if no player is within max_player_ball_distance.
        """

        # Get the ball's center point from its bounding box
        ball_position = get_center_of_bbox(ball_bbox)

        # Track the closest player found so far
        minimum_distance = 99999  # Start with a large number so any real distance beats it
        assigned_player = -1      # -1 means "no player assigned yet"

        for player_id, player in players.items():
            player_bbox = player['bbox']
            # In the bounding box format [x1, y1, x2, y2]:
            # player_bbox[0]  = x1  → left edge     (x coordinate)
            # player_bbox[-1] = y2  → bottom edge   (y coordinate)  
            # player_bbox[2]  = x2  → right edge    (x coordinate)
            # player_bbox[-1] = y2  → bottom edge   (y coordinate)  
            # Measure distance from the ball to the player's bottom-left foot
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)

            # Measure distance from the ball to the player's bottom-right foot
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)

            # Use whichever foot is closer to the ball
            distance = min(distance_left, distance_right)

            # Only consider this player if they're within the allowed range
            if distance < self.max_player_ball_distance:

                # If this player is closer than the previous closest, update the assignment
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player