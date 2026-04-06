from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    """
    Handles all object detection and tracking for players, referees, and the ball.

    Uses a YOLO model for detection and ByteTrack for maintaining consistent
    track IDs across frames. Also handles drawing all visual annotations
    (ellipses, triangles, ball control stats) onto the video frames.
    """

    def __init__(self, model_path):
        """
        Loads the YOLO model and initializes the ByteTrack tracker.

        Args:
            model_path: Path to the trained YOLO weights file (.pt)
        """
        self.model = YOLO(model_path)

        # ByteTrack assigns and maintains consistent IDs for each detected object
        # across frames — so player #5 stays #5 even if briefly occluded
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        """
        Adds a 'position' key to every track entry representing where the object is.

        - Ball position: center of its bounding box
        - Player/referee position: foot position (bottom-center of bounding box)
          since players stand on the ground and foot position is more accurate
          for field distance calculations.

        Args:
            tracks: Nested dict of {object: [{track_id: track_info}, ...]}

        Modifies tracks in-place by adding 'position' to each track entry.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']

                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)

                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        """
        Fills in missing ball positions between frames using linear interpolation.

        The ball is small and fast, so YOLO sometimes misses it for a few frames.
        Instead of having gaps, we interpolate — drawing a straight line between
        the last known and next known position to estimate where the ball was.

        Args:
            ball_positions: List of ball track dicts [{1: {'bbox': [...]}}, ...]
                            Missing frames have empty dicts {}.

        Returns:
            ball_positions: Same structure but with gaps filled in.
        """

        # Extract just the bbox values — missing frames become empty lists []
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]

        # Load into a DataFrame for easy interpolation
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate() fills gaps using linear interpolation between known values
        # bfill() handles any remaining gaps at the start of the video (backfill)
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Convert back to the original track dict format
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        """
        Runs YOLO detection on all frames in batches.

        Processing in batches (20 frames at a time) is more efficient than
        running detection one frame at a time — it makes better use of GPU memory.

        Args:
            frames: List of video frames (BGR images).

        Returns:
            detections: List of YOLO detection results, one per frame.
        """
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            # conf=0.1 is a low confidence threshold — we'd rather have false positives
            # than miss real detections (false negatives filtered later by tracking)
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects and tracks all players, referees, and the ball across every frame.

        Goalkeepers are remapped to the 'player' class since we don't compute
        separate stats for them. The ball is always assigned track_id=1 since
        there's only ever one ball.

        Results can be cached to disk and reloaded to avoid reprocessing
        the same video multiple times.

        Args:
            frames:          List of video frames (BGR images).
            read_from_stub:  If True, load cached results from disk instead of reprocessing.
            stub_path:       File path for saving/loading cached track data.

        Returns:
            tracks: {
                "players":  [{track_id: {"bbox": [...]}}, ...],  # one dict per frame
                "referees": [{track_id: {"bbox": [...]}}, ...],
                "ball":     [{1:        {"bbox": [...]}}, ...]
            }
        """

        # Load from cache if available — saves significant processing time
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names                          # {0: 'player', 1: 'referee', ...}
            cls_names_inv = {v: k for k, v in cls_names.items()}  # {'player': 0, 'referee': 1, ...}

            # Convert YOLO detections to Supervision format for ByteTrack compatibility
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Remap goalkeeper → player so they get tracked with the same class.
            # We don't compute separate stats for goalkeepers in this pipeline.
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Run ByteTrack — assigns persistent track IDs to each detection
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize empty dicts for this frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Store tracked players and referees (these have ByteTrack IDs)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Store ball separately — ball isn't tracked by ByteTrack, always gets id=1
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save to cache for future runs
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draws a colored ellipse at the player's feet and optionally a ID label above it.

        The ellipse is drawn at the bottom of the bounding box (feet level) and is
        wider than it is tall — giving a shadow/halo effect under the player.
        A small filled rectangle with the track ID number is drawn just below the ellipse.

        Args:
            frame:    Video frame to draw on.
            bbox:     Bounding box [x1, y1, x2, y2] of the player.
            color:    BGR color tuple for the ellipse and ID box.
            track_id: Optional player ID to display. If None, no label is drawn.

        Returns:
            frame: The frame with the ellipse (and optional label) drawn on it.
        """

        # Ellipse is drawn at the bottom-center of the bounding box (feet level)
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),  # Wide and flat ellipse
            angle=0.0,
            startAngle=-45,    # Partial ellipse — open at the top so it looks like a shadow
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # --- Draw track ID label ---
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20

            # Center the rectangle below the ellipse
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            # Nudge text left slightly for 3-digit IDs so it fits inside the box
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text on colored background
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        """
        Draws a small filled triangle above an object (used for the ball and
        to indicate which player currently has possession).

        The triangle points downward toward the object, acting like a marker/pin.

        Args:
            frame: Video frame to draw on.
            bbox:  Bounding box [x1, y1, x2, y2] of the object.
            color: BGR fill color for the triangle.

        Returns:
            frame: The frame with the triangle drawn on it.
        """

        # Place triangle above the top of the bounding box
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Triangle points: tip at object center, base 20px above
        triangle_points = np.array([
            [x, y],           # tip — points down toward the object
            [x - 10, y - 20], # base left
            [x + 10, y - 20]  # base right
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)  # Fill with team color
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)       # Black outline

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draws a running ball possession percentage for both teams on the frame.

        Shows what percentage of frames up to the current moment each team
        has had possession of the ball — updates live as the video plays.

        Args:
            frame:             Current video frame.
            frame_num:         Current frame index.
            team_ball_control: Numpy array where each entry is 1 or 2,
                               indicating which team had the ball that frame.

        Returns:
            frame: The frame with the possession stats overlay drawn on it.
        """

        # Draw semi-transparent white background box for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Only look at frames up to now — gives a running/cumulative possession stat
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # Count how many frames each team had possession
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Convert to percentages
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Composites all visual annotations onto every frame of the video.

        For each frame draws:
        - Colored ellipse + ID label under each player (color = their team color)
        - Red triangle over any player currently holding the ball
        - Cyan ellipse under each referee
        - Green triangle over the ball
        - Ball possession percentage overlay in the corner

        Args:
            video_frames:      List of raw video frames (BGR images).
            tracks:            Full tracking dict with team colors and ball possession flags.
            team_ball_control: Numpy array of team possession per frame (1s and 2s).

        Returns:
            output_video_frames: New list of annotated frames.
        """

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players — each gets their team color, plus a triangle if they have the ball
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # Default red if no team assigned
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw referees in cyan — no ID label needed
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball as a green triangle marker
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw running ball possession stats in the corner
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames