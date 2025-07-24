from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import cv2

# Load the trained YOLO model
model = YOLO(os.path.join("model", "indiabest.pt"))

# Predict on the input image
result = model.predict(
    source=os.path.join("model", "00000206.jpg"), 
    save=False
)

# Access the predictions
boxes = result[0].boxes  # Bounding box predictions
image_path = result[0].path  # Path to the original image
image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Draw bounding boxes on the image
for box in boxes:
    # Extract box coordinates and label
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    confidence = box.conf[0]  # Confidence score
    label = box.cls[0]  # Class label

    # Draw the box
    # if(confidence>0.3):
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # Put label and confidence
    text = f"Class: {label}, Conf: {confidence:.2f}"
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    


# Display the image with bounding boxes using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")  # Turn off axis for cleaner output
plt.title("YOLO Prediction with Bounding Boxes")

plt.show() 