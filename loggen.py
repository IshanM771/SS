import cv2
import os
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

# Load the saved pre-trained model
model = load_model('saved_model/s_model.h5')

# Load prebuilt classifier for Frontal Face detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Create or append to a log file
def assure_path_exists(path):
    if not path:
        print("Invalid path specified.")
        return

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            print(f"Error creating directory: {e}")

# Example usage
log_file_path = 'identification_log.txt'
log_file_path = os.path.abspath(log_file_path)
assure_path_exists(log_file_path)

# Method to preprocess images
def preprocess_image(image):
    # Convert single-channel grayscale image to three-channel RGB image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Resize image
    resized_image = cv2.resize(rgb_image, (224, 224))
    return resized_image.astype(np.float32) / 255.0


# Method to recognize faces and write to log file
def recognize_faces_and_write_log(frame, timestamp, log_file):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)  # default

    for (x, y, w, h) in faces:
        face_img = preprocess_image(gray[y:y+h, x:x+w])

        # Expand dimensions to match model input shape
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)

        prediction = model.predict(face_img)
        class_id = np.argmax(prediction)
        confidence = prediction[0, class_id]

        names = ["Ishan", "Amogh", "Niraj", "Harshada", "Unknown"]
        recognized_name = names[class_id]

        log_entry = f"{timestamp} - Recognized: {recognized_name} with confidence {confidence:.2f}\n"
        log_file.write(log_entry)

        cv2.rectangle(frame, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
        display_text = f"{recognized_name} ({confidence:.2f})"
        cv2.putText(frame, display_text, (x, y-40), font, 1, (255, 255, 255), 3)

    return frame

# Video file path
video_file = 'efg.mp4'

# Open the video file
cap = cv2.VideoCapture(video_file)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, 'a') as log_file:
        frame_with_boxes = recognize_faces_and_write_log(frame, timestamp, log_file)
        cv2.imshow('Frame', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
