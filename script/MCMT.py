import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO
from boxmot import DeepOCSORT, BoTSORT

# Function to set up video writers based on input streams
def create_video_writers(video_streams, output_filenames, codec='mp4v'):
    frame_width = int(video_streams[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_streams[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_streams[0].get(cv2.CAP_PROP_FPS))
    
    video_writers = [
        cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*codec), fps, (frame_width, frame_height))
        for filename in output_filenames
    ]
    return video_writers

# Initialize the tracker
reid_weights_path = Path('weights_ReID/pt/Market1501_clipreid_RN50_120.pt')
tracker = BoTSORT(
    reid_weights = reid_weights_path,  # which ReID model to use
    device = 'cuda:0',
    half = False
)

# Initialize YOLO model
model = YOLO('weights/yolov10x.pt')

# Open the input video
# Set up two camera feeds or video streams
camera_feeds = ['videos/05_1F_2024_9_2_8mins.mp4', 'videos/13_1F_2024_9_2_8mins.mp4']  # Replace with real-time camera URLs or video files
video_streams = [cv2.VideoCapture(feed) for feed in camera_feeds]

# Define output video filenames
output_filenames = [f'videos/tracking/output_camera_{i + 1}.mp4' for i in range(len(camera_feeds))]

# Create video writers
video_writers = create_video_writers(video_streams, output_filenames)

# Initialize frame count for each camera
frame_counts = [0 for _ in range(len(camera_feeds))]

# Confidence threshold for filtering low-confidence detections
CONFIDENCE_THRESHOLD = 0.5

while True:
    frames = []
    rets = []

    # Read frames from both camera feeds
    for stream in video_streams:
        ret, frame = stream.read()
        rets.append(ret)
        frames.append(frame)

    # If one of the cameras fails, exit the loop
    if not all(rets):
        print("One of the cameras has ended or failed.")
        break

    try:
        # Loop through each frame from both cameras
        for cam_idx, frame in enumerate(frames):
            # Update frame count for the current camera
            frame_counts[cam_idx] += 1    

            # Run YOLOv8 detection on the current frame
            results = model(frame, classes=[0], verbose=False)

            if len(results) >= 1:
                # Convert the detections to the required format: N X (x, y, x, y, conf, cls)
                dets = []
                for result in results:
                    for boxes in result.boxes:
                        conf = boxes.conf.item() # Get the confidence score
                        if conf >= CONFIDENCE_THRESHOLD: # Filter based on confidence threshold
                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = boxes.xyxy[0][0].item(), boxes.xyxy[0][1].item(), boxes.xyxy[0][2].item(), boxes.xyxy[0][3].item()
                            cls = boxes.cls.item()
                            dets.append([x1, y1, x2, y2, conf, int(cls)])
                dets = np.array(dets)

                # Check if there are any detections
                if dets.size > 0:
                    # Update the tracker with the detections
                    print("-----", tracker.update(dets, frame)) # --> M X (x, y, x, y, id, conf, cls, ind)
                # If no detections, make prediction ahead
                else:
                    dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
                    tracker.update(dets, frame) # --> M X (x, y, x, y, id, conf, cls, ind)

                # Plot results on the frame
                tracker.plot_results(frame, show_trajectories=True)
            
            # Write the frame with tracking annotations to the corresponding output video file
            video_writers[cam_idx].write(frame)

            # Print the frame count for the current camera to the terminal
            print(f"Camera {cam_idx + 1} - Processed Frame: {frame_counts[cam_idx]}")    

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Release video streams and close all windows
for stream in video_streams:
    stream.release()

for writer in video_writers:
    writer.release()

cv2.destroyAllWindows()