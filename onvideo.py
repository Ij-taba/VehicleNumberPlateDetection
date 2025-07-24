from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Load the trained YOLO model
model=YOLO("yolov8n.pt")
plate_dtetector = YOLO(os.path.join("model", "3rdtimetrain.pt"))

# Path to the input video
input_video_path = os.path.join("model", "WhatsApp Video 2025-01-22 at 4.26.33 AM.mp4")  # Replace with your video file path
# output_video_path = os.path.join("model", "output_video.mp4")  # Output video file path

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter for the output video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    print("open")
    ret, frame = cap.read()
    if not ret:
        
        break  # Exit the loop if no frame is read (end of video)
    
    # frame = cv2.fastNlMeansDenoisingColored(frame, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    # Predict on the current frame
    # frame = cv2.filter2D(frame, -1, sharpening_kernel)
    # smooth_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply mild sharpening filter
    # sharpening_kernel = np.array([[0, -0.5, 0],
    #                                [-0.5, 3, -0.5],
    #                                [0, -0.5, 0]])
    # frame = cv2.filter2D(frame, -1, sharpening_kernel)
    desired_width = 600
    desired_height = 600
    frame= cv2.resize(frame, (desired_width, desired_height))
    detections_=[]
    # Convert frame to black-and-white (grayscale)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections=model.predict(frame)[0]
    for detection in detections.boxes.data.tolist():
        x1,y1,x2,y2,c,cl=detection
        if(cl==3):
            detections_.append([x1,y1,x2,y2,c])

    results = model.predict(source=frame, save=False, verbose=False)
    boxes = results[0].boxes  # Bounding box predictions

    # frame = cv2.fastNlMeansDenoisingColored(frame, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    # Draw bounding boxes on the frame
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        label = int(box.cls[0])  # Class label
        print("yyy")
        # Draw the bounding box if confidence > threshold
        if confidence > 0.1:  # Adjust confidence threshold as needed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"Class: {label}, Conf: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("frame",frame)
    cv2.waitKey(5)  
# cv2.imwrite("ff1",frame)        
    # Write the processed frame to the output video
    # out.write(frame)

# Release resources
print("complete")
cap.release()
# out.release()

# print(f"Processed video saved at: {output_video_path}")

