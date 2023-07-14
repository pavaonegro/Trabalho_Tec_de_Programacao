
from datetime import timedelta
import cv2
import numpy as np
import os

# Save frames every n seconds
FRAME_SAVE_INTERVAL = 1

def extract_frames(video_file):
    # Create a folder to save the frames
    frames_folder = "frames"
    if not os.path.isdir(frames_folder):
        os.makedirs(frames_folder)

    # Read the video file
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame extraction interval
    interval = int(fps * FRAME_SAVE_INTERVAL)

    # Initialize variables
    count = 0
    duration = timedelta(seconds=1 / fps)

    while True:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)

        # Read the frame
        is_read, frame = cap.read()
        if not is_read:
            break

        # Save the frame
        frame_path = os.path.join(frames_folder, f"frame{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Calculate progress percentage
        progress = count / frame_count * 100
        print(f"Progress: {progress:.2f}%")

        # Update the frame count and duration
        count += interval
        duration += timedelta(seconds=FRAME_SAVE_INTERVAL)

        # Skip frames to maintain the desired interval
        while duration.total_seconds() > cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:
            is_read = cap.grab()
            if not is_read:
                break

    cap.release()

    return frames_folder

def motion_detection(frames_folder):
    # Create a folder to save the motion detected frames
    motion_detected_folder = "motion_detected_frames"
    if not os.path.isdir(motion_detected_folder):
        os.makedirs(motion_detected_folder)

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    for frame_file in sorted(os.listdir(frames_folder)):
        # Read frame from file
        frame = cv2.imread(os.path.join(frames_folder, frame_file))

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Perform morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            # Draw bounding box around the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the motion-detected frame
        motion_detected_path = os.path.join(motion_detected_folder, frame_file)
        cv2.imwrite(motion_detected_path, frame)

    print(f"Motion detected frames saved to: {motion_detected_folder}")

if __name__ == "__main__":
    video_file = r"C:\Users\GCORREAA\Pictures\trabalho_tec_programacao_gustavo_manoela\VisitasAbelhas.mp4"
    frames_folder = extract_frames(video_file)
    motion_detection(frames_folder)