import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('/home/jovyan/Object Detection/Object Detection/runs/detect/train/weights/best.pt')

# Open a video file
video_path = '/home/jovyan/Object Detection/Object Detection/Video 111 double0.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    H, W, _ = frame.shape
    if not ret:
        break

    # Make predictions
    
    results = model(frame)
    for result in results:
        result.render()


    # Render results on the frame (bounding boxes, labels, etc.)
    # results.render()  # This function modifies the frame in-place

    # Display the frame
    cv2.imshow('YOLOv8 Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit the loop
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

