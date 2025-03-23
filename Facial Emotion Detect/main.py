import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# Step 1: Load and Preprocess the Data
def load_and_preprocess_data():
    # Load the FER-2013 dataset
    data = pd.read_csv('fer2013.csv')  # Ensure the file is in your working directory

    X = []
    y = []

    for index, row in data.iterrows():
        pixels = np.array(row['pixels'].split(), dtype='float32')
        X.append(pixels.reshape(48, 48, 1))  # Reshape to 48x48x1 (grayscale)
        y.append(row['emotion'])

    X = np.array(X) / 255.0  # Normalize pixel values to [0, 1]
    y = np.array(y)

    # Convert labels to one-hot encoding
    num_classes = 7
    y = tf.keras.utils.to_categorical(y, num_classes)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Step 2: Build the Model
def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Step 3: Train the Model
def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model, history

# Step 4: Real-Time Webcam Prediction
def webcam_prediction(model):
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Map class indices to emotion labels
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = gray[y:y+h, x:x+w]

            # Resize the face ROI to 48x48 (input size for the model)
            resized_face = cv2.resize(face_roi, (48, 48))
            resized_face = resized_face.reshape(1, 48, 48, 1) / 255.0

            # Predict the emotion
            prediction = model.predict(resized_face)
            predicted_class = np.argmax(prediction)
            emotion = emotion_labels[predicted_class]

            # Draw a rectangle around the face and display the predicted emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Facial Emotion Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Main Function
def main():
    # Step 1: Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Step 2: Build the model
    model = build_model()

    # Step 3: Train the model
    model, history = train_model(model, X_train, y_train, X_test, y_test)

    # Step 4: Save the model
    model.save('facial_emotion_recognition_model.h5')

    # Step 5: Real-time webcam prediction
    webcam_prediction(model)

if __name__ == "__main__":
    main()