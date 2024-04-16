import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from mtcnn import MTCNN

def load_data_and_labels(directory):
    images, labels = [], []
    label_mapping = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            label = filename.split('.')[0]  # Extract label from filename
            if label not in label_mapping:
                label_mapping[label] = len(label_mapping)
            
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))  # Resize image to match model input size
            image = image.astype(np.float32) / 255.0  # Normalize pixel values
            images.append(image)
            labels.append(label_mapping[label])
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, label_mapping




def preprocess_images(images):
    resized_images = [cv2.resize(image, (224, 224)) for image in images]
    return np.array(resized_images) / 255.0

def train_model():
    # Load and preprocess data
    images, labels, label_mapping = load_data_and_labels('training_data')
    images = preprocess_images(images)
    
    print("Shape of images:", images.shape)
    print("Type of images:", images.dtype)
    print("Min pixel value:", np.min(images), "Max pixel value:", np.max(images))
    
    # Split data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    print("Number of training images:", len(train_images))
    print("Number of validation images:", len(val_images))
    
    print("Shape of train_images:", train_images.shape)
    print("Type of train_images:", train_images.dtype)
    
    print("Shape of train_labels:", train_labels.shape)
    print("Type of train_labels:", train_labels.dtype)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(train_images)
    
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(label_mapping), activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('saved_model/best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    
    # Train the model
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                        validation_data=(val_images, val_labels),
                        epochs=50,
                        callbacks=[es, mc])
    
    # Save the model
    model.save('saved_model/s_model.h5')
    
    return history, label_mapping


if __name__ == "__main__":
    train_model()
