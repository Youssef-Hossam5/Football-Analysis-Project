import cv2


def read_video(video_path):
    """
    Reads a video file and returns all its frames as a list.

    Args:
        video_path: Path to the input video file.

    Returns:
        frames: List of frames (BGR images) in order.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        # ret = False when there are no more frames to read
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames


def save_video(output_video_frames, output_video_path):
    """
    Saves a list of frames as a video file.

    Uses XVID codec at 24fps. Output dimensions are taken automatically
    from the first frame so no manual sizing is needed.

    Args:
        output_video_frames: List of frames (BGR images) to write.
        output_video_path:   Path to save the output video file.
    """
    # XVID is a widely supported compressed video codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,  # Frame rate — must match the original video for correct playback speed
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])  # (width, height)
    )

    for frame in output_video_frames:
        out.write(frame)

    # Release the writer to flush and properly close the file
    out.release()