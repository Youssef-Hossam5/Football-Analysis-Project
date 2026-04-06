from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # -----------------------------------------------------------------------
    # 1. LOAD VIDEO
    # -----------------------------------------------------------------------
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # -----------------------------------------------------------------------
    # 2. DETECT & TRACK OBJECTS (Players, Referees, Ball)
    # Uses YOLO for detection and ByteTrack for consistent IDs across frames.
    # read_from_stub=True loads cached results to skip reprocessing.
    # -----------------------------------------------------------------------
    tracker = Tracker('models/yolov5xu.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    # Add raw pixel position (center for ball, feet for players) to every track entry
    tracker.add_position_to_tracks(tracks)

    # -----------------------------------------------------------------------
    # 3. ESTIMATE & COMPENSATE FOR CAMERA MOVEMENT
    # The camera pans to follow the action — this detects how much it moved
    # each frame using optical flow, then subtracts that movement from all
    # object positions so they're relative to the field, not the camera.
    # -----------------------------------------------------------------------
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # -----------------------------------------------------------------------
    # 4. PERSPECTIVE TRANSFORM — PIXELS → REAL WORLD METERS
    # The camera angle distorts distances (far = smaller). This maps pixel
    # coordinates to actual field coordinates in meters for accurate measurements.
    # -----------------------------------------------------------------------
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # -----------------------------------------------------------------------
    # 5. INTERPOLATE MISSING BALL POSITIONS
    # YOLO sometimes misses the ball for a few frames. Linear interpolation
    # fills those gaps so ball tracking is continuous throughout the video.
    # -----------------------------------------------------------------------
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # -----------------------------------------------------------------------
    # 6. CALCULATE PLAYER SPEED & DISTANCE
    # Uses real-world (transformed) positions to compute speed in km/h and
    # cumulative distance in meters for each player across the video.
    # -----------------------------------------------------------------------
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # -----------------------------------------------------------------------
    # 7. ASSIGN PLAYERS TO TEAMS
    # Samples jersey colors from the first frame, clusters them into 2 teams,
    # then classifies every player in every frame to a team.
    # -----------------------------------------------------------------------
    team_assigner = TeamAssigner()

    # Use the first frame to establish the two team colors
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # Classify every player in every frame and store their team + team color
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # -----------------------------------------------------------------------
    # 8. ASSIGN BALL POSSESSION
    # For each frame, finds the player closest to the ball (within threshold).
    # Records which team has possession each frame for the running stats overlay.
    # If no player is close enough, the previous frame's possession is carried forward.
    # -----------------------------------------------------------------------
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            # Mark the player as having the ball (used to draw triangle indicator)
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # No one has the ball — carry forward the last known possession
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # -----------------------------------------------------------------------
    # 9. DRAW ANNOTATIONS ONTO FRAMES
    # Layers all visual elements onto the video frames in order:
    #   - Player ellipses + ID labels + ball possession triangles
    #   - Camera movement overlay (x/y values)
    #   - Speed and distance stats below each player
    # -----------------------------------------------------------------------
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # -----------------------------------------------------------------------
    # 10. SAVE OUTPUT VIDEO
    # -----------------------------------------------------------------------
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()