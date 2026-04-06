import pickle
import cv2
import numpy as np
import os
import sys

sys.path.append('../')
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    """
    Estimates how much the camera is physically moving between video frames.

    In sports broadcasts, the camera often pans left/right or up/down to follow the action.
    This class detects that movement so we can "cancel it out" when tracking players/objects —
    otherwise a player standing still would appear to move just because the camera moved.

    How it works:
        1. Pick trackable points (corners/edges) in the first frame — only on the left/right borders
           of the frame where there's usually stable background (not players running around).
        2. For each new frame, use optical flow to see where those points moved.
        3. If the points moved significantly, that movement is the camera moving.
        4. Store [x, y] camera movement for every frame.
    """

    def __init__(self, frame):
        """
        Sets up the camera movement estimator using the first video frame.

        Args:
            frame: The first frame of the video (BGR color image from OpenCV).
        """

        # Minimum pixel distance a feature point must move to count as real camera movement.
        # Small movements (< 5px) are ignored — they're likely just noise or compression artifacts.
        self.minimum_distance = 5

        # --- Lucas-Kanade Optical Flow Settings ---
        # These control how OpenCV tracks points from one frame to the next.
        self.lk_params = dict(
            winSize=(15, 15),       # Search window around each point — 15x15 pixels
            maxLevel=2,             # Use image pyramids (2 levels) to catch both small & large movements
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,     # Stop after 10 iterations max
                0.03    # Or stop when accuracy reaches 0.03 (whichever comes first)
            )
        )

        # --- Feature Detection Mask ---
        # We only want to track points on the LEFT and RIGHT edges of the frame.
        # This is because players are usually in the center — we don't want to track them,
        # we want to track the background/field markings which only move when the camera moves.
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)  # Start with all zeros (ignore everything)
        mask_features[:, 0:20] = 1       # Activate left edge (columns 0 to 19)
        mask_features[:, 900:1050] = 1   # Activate right edge (columns 900 to 1049)

        # --- Feature Detection Settings ---
        # These are passed to cv2.goodFeaturesToTrack() which finds the best points to track.
        self.features = dict(
            maxCorners=100,         # Find at most 100 trackable points
            qualityLevel=0.3,       # Only keep points with at least 30% of the best corner quality
            minDistance=3,          # Points must be at least 3 pixels apart from each other
            blockSize=7,            # Neighborhood size used to compute corner strength
            mask=mask_features      # Only look for points in our masked (edge) regions
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Calculates how much the camera moved (in pixels) for each frame in the video.

        Instead of recalculating every time, results can be saved to a file ("stub") and
        loaded later — useful when processing the same video multiple times.

        Args:
            frames:           List of video frames (BGR images).
            read_from_stub:   If True, try to load previously saved results from disk.
            stub_path:        File path to save/load the cached results.

        Returns:
            camera_movement:  A list of [x, y] values — one per frame — showing how many
                              pixels the camera moved horizontally (x) and vertically (y).
                              Frame 0 is always [0, 0] since there's nothing to compare it to.
        """

        # --- Load from cache if available ---
        # This saves time when re-running on the same video
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Start with zero movement for every frame.
        # Using a list comprehension here (NOT [[0,0]] * len) to ensure each entry
        # is an independent list — multiplying a list creates shared references, which is a bug.
        camera_movement = [[0, 0] for _ in frames]

        # Convert the first frame to grayscale — optical flow works on single-channel images
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        # Find the initial set of trackable feature points in frame 0
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # --- Process Each Frame ---
        for frame_num in range(1, len(frames)):

            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # If we have no features to track (edge case), skip this frame
            if old_features is None:
                old_gray = frame_gray.copy()
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                continue

            # Track where the old feature points moved to in the new frame
            # new_features = where each point ended up
            # status = 1 if the point was successfully tracked, 0 if lost
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            # If tracking completely failed, skip this frame
            if new_features is None:
                old_gray = frame_gray.copy()
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                continue

            # --- Find the Largest Movement Among All Tracked Points ---
            # We use the point that moved the MOST as our best estimate of camera movement.
            # (If the camera panned, all background points should move roughly the same amount.)
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for new, old in zip(new_features, old_features):
                new_point = new.ravel()   # Flatten from [[x, y]] to [x, y]
                old_point = old.ravel()

                # Euclidean distance between old and new position
                distance = measure_distance(new_point, old_point)

                if distance > max_distance:
                    max_distance = distance
                    # Record x and y shift separately so we know direction of movement
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_point, new_point)

            # --- Only Record Movement if It's Above the Noise Threshold ---
            # Small movements (< minimum_distance) are probably just video noise, not real camera movement
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]

                # Re-detect feature points in the current frame for the next iteration.
                # This keeps tracking fresh and handles cases where old points go off-screen.
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # The current frame becomes the "old" frame for the next iteration
            old_gray = frame_gray.copy()

        # --- Save results to cache file if a path was given ---
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjusts every tracked object's position to remove the camera's influence.

        Without this, a player standing perfectly still would appear to move across the screen
        just because the camera panned. This function subtracts the camera's movement from
        every object's position so that positions are relative to the field, not the camera lens.

        Args:
            tracks:                     The full tracking data — nested dict of
                                        {object_type: [{track_id: track_info}, ...]}
            camera_movement_per_frame:  List of [x, y] camera movement per frame
                                        (output of get_camera_movement).

        Modifies tracks in-place by adding a 'position_adjusted' key to each track entry.
        """

        for object_type, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():

                    # Original position of the object in this frame (in screen pixels)
                    position = track_info['position']

                    # How much the camera moved in this frame
                    camera_movement = camera_movement_per_frame[frame_num]

                    # Subtract camera movement to get the "real" position on the field
                    # If camera moved right by 10px, every object appears 10px to the right —
                    # so we subtract 10 to get back to the field-relative position.
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1]
                    )

                    # Store adjusted position alongside the original
                    tracks[object_type][frame_num][track_id]['position_adjusted'] = position_adjusted

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Draws a live overlay on each frame showing the current camera movement values.

        Renders a semi-transparent white box in the top-left corner with the X and Y
        movement values printed inside — useful for debugging and visualizing the output.

        Args:
            frames:                     List of video frames (BGR images).
            camera_movement_per_frame:  List of [x, y] camera movement per frame.

        Returns:
            output_frames: New list of frames with the camera movement overlay drawn on each.
        """

        output_frames = []

        for frame_num, frame in enumerate(frames):

            # Work on a copy so we don't modify the original frames
            frame = frame.copy()

            # --- Draw semi-transparent white background box ---
            # This makes the text readable regardless of what's behind it
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)  # Solid white rectangle
            alpha = 0.6  # 60% overlay, 40% original frame showing through
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            del overlay
            # --- Write X and Y movement values as text ---
            x_movement, y_movement = camera_movement_per_frame[frame_num]

            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",   # Rounded to 2 decimal places
                (10, 30),                                  # Position: 10px from left, 30px from top
                cv2.FONT_HERSHEY_SIMPLEX,
                1,          # Font scale
                (0, 0, 0),  # Black text
                3           # Line thickness
            )
            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),   # Second line sits 30px below the first
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3
            )

            output_frames.append(frame)
            frames[frame_num] = None

        return output_frames