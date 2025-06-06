import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
import traceback
import time
import urllib.request
from sklearn.utils.class_weight import compute_class_weight

# --- Step 1: Define paths and constants ---
# IMPORTANT: Adjust these paths to match YOUR exact directory structure.
# train_dir and test_dir should point to the 'train' and 'test' subfolders within your dataset.
# If your dataset is structured like 'dataset/anger', 'dataset/happy' etc.,
# and not split into 'train'/'test' at the top level, you'll need to
# adjust the tf.keras.preprocessing.image_dataset_from_directory calls below
# or manually create 'train' and 'test' folders and split your data.
train_dir = "C:/Users/divye/EmotionDetection/dataset/train" # Example path, adjust as needed
test_dir = "C:/Users/divye/EmotionDetection/dataset/test"   # Example path, adjust as needed

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32
NUM_CLASSES = 7  # Based on your 7 emotion folders (anger, contempt, disgust, fear, happy, sadness, surprise)
EPOCHS = 100
MODEL_PATH = "C:/Users/divye/EmotionDetection/emotion_model.keras" # Path for a standard model
NEW_MODEL_PATH = "C:/Users/divye/EmotionDetection/emotion_model_improved.keras" # Path for the improved model

# Debug directory for saving frames if needed (currently commented out in loop)
DEBUG_DIR = "C:/Users/divye/EmotionDetection/debug_frames"
os.makedirs(DEBUG_DIR, exist_ok=True)

# --- Step 2: Load and preprocess the dataset using tf.data.Dataset ---
try:
    # Load datasets directly from directory
    # If your dataset is not split into 'train' and 'test' folders at the top level,
    # and instead all emotion folders are directly under a 'dataset' folder,
    # you might need to use tf.keras.utils.image_dataset_from_directory with validation_split
    # or reorganize your dataset.
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        label_mode='categorical', # Labels will be one-hot encoded (e.g., [0,1,0,0,0,0,0])
        shuffle=True
    )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        label_mode='categorical',
        shuffle=False
    )

    # Print detected class names to confirm correct loading order
    print("Class indices (alphabetical by folder name):", train_dataset.class_names)
    # The order here is crucial for mapping model output to emotion labels in the real-time detection.
    # Make sure emotion_labels in real-time section matches this order.

    # Compute class weights to handle imbalance in dataset (if some emotion classes have fewer images)
    labels = []
    for _, label_batch in train_dataset:
        labels.extend(np.argmax(label_batch, axis=1)) # Convert one-hot labels to integer labels
    labels = np.array(labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = dict(enumerate(class_weights))
    print("Computed class weights:", class_weights_dict)

    # Define data augmentation pipeline using Keras preprocessing layers
    # These layers apply transformations on the CPU before passing to the model.
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.15),  # Rotate by up to 15% of 2*pi radians (approx 54 degrees)
        RandomTranslation(height_factor=0.1, width_factor=0.1), # Translate by up to 10% of image height/width
        RandomZoom(height_factor=(-0.2, 0.2)),  # Zoom by -20% to +20%
        RandomContrast(factor=0.2) # Adjust contrast randomly by up to 20%
    ])

    # Function to normalize image pixel values from [0, 255] to [0, 1]
    def normalize_image(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Apply augmentation and preprocessing to datasets
    # AUTOTUNE optimizes the number of parallel calls and prefetching.
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    ).map(normalize_image, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    test_dataset = test_dataset.map(normalize_image, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

except Exception as e:
    print("Error during dataset loading and preprocessing:", str(e))
    traceback.print_exc()
    sys.exit(1)

# --- Step 3: Build the improved CNN model ---
def build_improved_model():
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3), # Increased dropout

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 4
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3), # Can increase this for very deep networks

        Flatten(), # Convert 3D feature maps to 1D vector

        # Dense Layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5), # Higher dropout for dense layers

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # Output Layer
        Dense(NUM_CLASSES, activation='softmax') # Output probabilities for each emotion class
    ])

    model.compile(optimizer='adam', # Adam optimizer with default learning rate (usually 0.001)
                  loss='categorical_crossentropy', # Appropriate for one-hot encoded labels
                  metrics=['accuracy'])

    return model

# --- Step 4: Alternative: Transfer Learning with VGG16 (for RGB images typically) ---
# Note: VGG16 expects 3-channel (RGB) images. Your dataset is grayscale.
# A Lambda layer is used to convert grayscale to RGB for VGG16.
def build_transfer_learning_model():
    # Load VGG16 pre-trained on ImageNet, excluding its top (classification) layers
    # input_tensor=Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)) because VGG16 expects RGB
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    base_model.trainable = False # Freeze the pre-trained layers

    model = Sequential([
        # Lambda layer to convert grayscale (1 channel) to RGB (3 channels) for VGG16 input
        tf.keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x), input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        base_model, # The VGG16 model as a feature extractor
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# --- Step 5: Train the model (or load if already trained) ---
try:
    if os.path.exists(NEW_MODEL_PATH):
        print(f"Loading the improved model from {NEW_MODEL_PATH}...")
        model = load_model(NEW_MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("Improved model not found. Building and training a new model...")
        model = build_improved_model() # You could change this to build_transfer_learning_model() if desired
        model.summary()

        # Only train if this script is run directly (not imported as a module)
        if __name__ == "__main__":
            print("Starting model training...", flush=True)
            history = model.fit(
                train_dataset,
                steps_per_epoch=len(train_dataset), # Ensure training covers all batches
                epochs=EPOCHS,
                validation_data=test_dataset,
                validation_steps=len(test_dataset), # Ensure validation covers all batches
                class_weight=class_weights_dict # Apply computed class weights
            )

            model.save(NEW_MODEL_PATH)
            print(f"Improved model trained and saved at {NEW_MODEL_PATH}", flush=True)

            # Plot training history (accuracy and loss over epochs)
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.show() # Display the plots

except Exception as e:
    print("Error during model training/loading:", str(e), flush=True)
    traceback.print_exc()
    plt.show() # Attempt to show plot if an error occurred during plotting itself
    sys.exit(1)

# --- Step 6: Real-time emotion detection ---
try:
    # Download Haar Cascade file if not present in the project directory
    face_cascade_path = "C:/Users/divye/EmotionDetection/haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_path):
        print(f"Haar Cascade file not found. Downloading to {face_cascade_path}...", flush=True)
        # Ensure you have internet connection for this download
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", face_cascade_path)
        print("Haar Cascade file downloaded successfully.", flush=True)
    else:
        print("Haar Cascade file already exists.", flush=True)

    print("Loading Haar Cascade classifier...", flush=True)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("Error: Failed to load Haar Cascade classifier. Check the path to", face_cascade_path, flush=True)
        sys.exit(1)
    print("Haar Cascade classifier loaded successfully.", flush=True)

    # Download DNN face detector files if not present
    dnn_prototxt_path = "C:/Users/divye/EmotionDetection/deploy.prototxt"
    dnn_model_path = "C:/Users/divye/EmotionDetection/res10_300x300_ssd_iter_140000.caffemodel"
    if not os.path.exists(dnn_prototxt_path):
        print("Downloading DNN prototxt file...", flush=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", dnn_prototxt_path)
        print("DNN prototxt file downloaded successfully.", flush=True)
    if not os.path.exists(dnn_model_path):
        print("Downloading DNN model file...", flush=True)
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/master/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            dnn_model_path
        )
        print("DNN model file downloaded successfully.", flush=True)

    print("Loading DNN face detector as fallback...", flush=True)
    net = cv2.dnn.readNetFromCaffe(dnn_prototxt_path, dnn_model_path)
    print("DNN face detector loaded successfully.", flush=True)

    # Emotion labels (ensure this order matches your model's class indices)
    # The order will be alphabetical based on your dataset folder names
    emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise'] # Check class_names from train_dataset for exact order

    print("Starting real-time emotion detection... Press 'q' to quit.", flush=True)

    # Attempt to open webcam with multiple indices
    cap = None
    for i in range(3): # Try indices 0, 1, 2
        print(f"Attempting to open webcam with index {i}...", flush=True)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Webcam opened successfully with index {i}.", flush=True)
            break
        else:
            print(f"Error: Could not open webcam with index {i}.", flush=True)
    
    if not cap or not cap.isOpened():
        print("Error: Could not open any webcam. Exiting...", flush=True)
        sys.exit(1)

    loop_count = 0
    use_dnn = False # Flag to switch to DNN detector after 10 failed Haar cascade attempts

    while True:
        loop_count += 1
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print(f"Loop {loop_count}: Failed to capture video frame. Ensure your webcam is working and not in use by another application.", flush=True)
            # Display a placeholder if frame capture fails
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Error! Check Connection.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Emotion Detection', error_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(1) # Wait a bit before retrying capture
            continue # Skip processing for this failed frame

        # Print for debugging frame capture success
        print(f"Loop {loop_count}: Captured frame successfully. Shape: {frame.shape}", flush=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"Loop {loop_count}: Converted frame to grayscale.", flush=True)

        faces = [] # Initialize faces as an empty list for this frame

        if not use_dnn:
            # Haar Cascade detection
            haar_detections = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(20, 20)
            )
            print(f"Loop {loop_count}: Haar Cascade detected {len(haar_detections)} faces.", flush=True)

            # Add Haar detections to the faces list
            for (x, y, w, h) in haar_detections:
                faces.append((x, y, w, h))

            # Switch to DNN if Haar Cascade consistently fails
            if len(faces) == 0 and loop_count % 10 == 0 and loop_count > 0: # Check every 10 loops if no faces
                print("Haar Cascade failed to detect faces repeatedly. Switching to DNN face detector.", flush=True)
                use_dnn = True
                # No need to reset loop_count here, use_dnn flag handles the switch

        if use_dnn or (len(faces) == 0 and loop_count % 10 != 0): # Try DNN if flag set OR if Haar failed in current loop (but not for 10 loops yet)
            # DNN face detection
            (h_frame, w_frame) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5: # Confidence threshold for DNN detection
                    box = detections[0, 0, i, 3:7] * np.array([w_frame, h_frame, w_frame, h_frame])
                    (x, y, x2, y2) = box.astype("int")
                    # Ensure bounding box is within frame boundaries
                    x, y, x2, y2 = max(0, x), max(0, y), min(w_frame, x2), min(h_frame, y2)
                    w_dnn = x2 - x
                    h_dnn = y2 - y
                    if w_dnn > 0 and h_dnn > 0: # Ensure valid dimensions
                         faces.append((x, y, w_dnn, h_dnn))

            print(f"Loop {loop_count}: DNN detected {len(faces)} faces.", flush=True)


        # Process each detected face (from either Haar or DNN)
        if len(faces) == 0:
            print(f"Loop {loop_count}: No faces detected. Ensure your face is visible, well-lit, and not too far/close to the camera.", flush=True)
            # Display the frame even without faces
            cv2.imshow('Emotion Detection', frame)
        else:
            for (x, y, w, h) in faces:
                # Extract Face ROI and Preprocess for Model
                face_roi = gray[y:y+h, x:x+w]
                # Check for valid ROI before resizing
                if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                    print(f"Loop {loop_count}: Invalid face_roi detected, skipping prediction.", flush=True)
                    continue # Skip this face if ROI is invalid

                face_roi_resized = cv2.resize(face_roi, (IMG_HEIGHT, IMG_WIDTH))
                face_roi_normalized = face_roi_resized / 255.0
                face_roi_input = face_roi_normalized.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1) # Reshape for model input

                # Make Emotion Prediction
                prediction = model.predict(face_roi_input)
                emotion = emotion_labels[np.argmax(prediction)]
                print(f"Loop {loop_count}: Predicted emotion: {emotion}", flush=True)

                # Draw bounding box and text on the original color frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Emotion Detection', frame) # Display the frame with annotations

        loop_time = time.time() - start_time
        print(f"Loop {loop_count}: Total loop time: {loop_time:.3f} seconds", flush=True)

        # Exit condition: Press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Exiting real-time detection.", flush=True)
            break

    # Release resources after the loop
    cap.release()
    cv2.destroyAllWindows()
    print("Real-time detection stopped.", flush=True)

except Exception as e:
    print("Error during real-time detection:", str(e), flush=True)
    traceback.print_exc()
    if 'cap' in locals() and cap.isOpened(): # Only release if cap was successfully opened
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(1)

# Ensure this part is only executed when the script is run directly
if __name__ == "__main__":
    print("Script finished execution.", flush=True)
