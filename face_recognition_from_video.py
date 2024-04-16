import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os

# Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def face_recognition_from_video(file):
    # Load the saved pre-trained model
    model_path = 'saved_model/s_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("Model not found. Please run 'training.py' to train the model.")
        exit()

    # Initialize MTCNN for face detection
    detector = MTCNN()

    # Initialize and start the video frame capture from a video clip
    # video_file = 'efg.mp4'
    video_file = file

    cam = cv2.VideoCapture(video_file)

    # Looping starts here
    while True:
        # Read the video frame
        ret, im = cam.read()

        if not ret:
            print("Video ended or not found.")
            break

        # Detect faces using MTCNN
        faces = detector.detect_faces(im)

        # For each face in faces, we will start predicting using the pre-trained model
        for face in faces:
            x, y, w, h = face['box']

            # Create rectangle around the face
            cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

            # Preprocess the face image before prediction
            face_roi = im[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, (224, 224)) / 255.0
            face_rgb = np.reshape(face_rgb, (1, 224, 224, 3))

            # Make prediction using the pre-trained model
            prediction = model.predict(face_rgb)

            # Set the name according to prediction
            if prediction[0][0] >= 0.5:
                name = "Person 1"
            else:
                name = "Unknown"

            # Set rectangle around face and name of the person
            cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
            cv2.putText(im, name, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # Display the video frame with the bounded rectangle
        cv2.imshow('im', im)

        # Press 'q' to close the program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Terminate video
    cam.release()

    # Close all windows
    cv2.destroyAllWindows()
