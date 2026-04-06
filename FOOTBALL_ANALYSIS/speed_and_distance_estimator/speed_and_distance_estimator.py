import cv2
import sys
sys.path.append('../')
from utils import measure_distance, get_foot_position


class SpeedAndDistance_Estimator:
    """
    Calculates and displays the real-world speed and total distance covered
    by each player throughout the video.

    Works by comparing a player's field position every N frames (a "window"),
    computing how far they moved, and dividing by the time elapsed.

    Note: Ball and referees are skipped — we only track players.
    """

    def __init__(self):
        # How many frames to look back/forward when calculating speed.
        # Instead of comparing every single frame (noisy), we sample every 5 frames.
        self.frame_window = 5

        # Video frame rate — needed to convert frame counts into real time (seconds)
        self.frame_rate = 24

    def add_speed_distance_to_tracks(self, tracks):
        """
        Calculates speed (km/h) and cumulative distance (meters) for every player
        and writes those values directly into the tracks dictionary.

        Works in windows of `frame_window` frames at a time:
            - Finds where a player was at the START of the window
            - Finds where they were at the END of the window
            - Computes distance covered and time elapsed → speed
            - Stamps that speed/distance onto every frame inside the window

        Args:
            tracks: Nested dict of {object: [{track_id: track_info}, ...]}
                    Each track_info must have a 'position_transformed' key
                    (real-world field coordinates, not raw pixel coordinates).

        Modifies tracks in-place by adding 'speed' and 'distance' to each track entry.
        """

        # Keeps a running total of distance covered per player across the whole video
        total_distance = {}

        for object, object_tracks in tracks.items():

            # Skip ball and referees — we only care about outfield players
            if object == 'ball' or object == 'referees':
                continue

            number_of_frames = len(object_tracks)

            # Step through the video in chunks of frame_window (e.g. every 5 frames)
            for frame_num in range(0, number_of_frames, self.frame_window):

                # The last frame of this window — clamp to avoid going out of bounds
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():

                    # If this player isn't present in the last frame of the window,
                    # they may have left the frame — skip them
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Get the player's real-world field position at start and end of window
                    # 'position_transformed' is in meters on the field (not screen pixels)
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # Skip if position data is missing (player may be partially off-frame)
                    if start_position is None or end_position is None:
                        continue

                    # --- Calculate Speed ---
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate  # seconds
                    speed_metres_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_metres_per_second * 3.6  # convert m/s → km/h

                    # --- Accumulate Total Distance ---
                    # Initialize nested dict entries if this is the first time seeing this object/player
                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    # --- Stamp speed and distance onto every frame in this window ---
                    # All frames in the window get the same speed value (it's an average over the window)
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        """
        Draws each player's current speed and total distance covered
        directly onto the video frames, positioned just below their feet.

        Args:
            frames: List of video frames (BGR images).
            tracks: Tracks dict after add_speed_distance_to_tracks() has been run
                    (so each track entry already has 'speed' and 'distance' values).

        Returns:
            output_frames: New list of frames with speed/distance text drawn on each player.
        """

        output_frames = []

        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():

                # Skip ball and referees — only draw stats for players
                if object == 'ball' or object == 'referees':
                    continue

                for _, track_info in object_tracks[frame_num].items():

                    # Only draw if speed data exists for this player in this frame
                    if 'speed' not in track_info:
                        continue

                    speed = track_info.get('speed', None)
                    distance = track_info.get('distance', None)

                    if speed is None or distance is None:
                        continue

                    # Position the text just below the player's feet
                    # get_foot_position returns the bottom-center of the bounding box
                    bbox = track_info['bbox']
                    position = list(get_foot_position(bbox))
                    position[1] += 40   # Push text 40px below feet so it doesn't overlap the player

                    # Convert to int tuple — cv2.putText requires integer coordinates
                    position = tuple(map(int, position))

                    # Draw speed on the first line
                    cv2.putText(frame, f"{speed:.2f} km/h", position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    # Draw total distance 20px below the speed text
                    cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames