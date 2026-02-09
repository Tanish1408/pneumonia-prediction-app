import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & DATASET
# ==========================================
# NOTE: The following lines are for Google Colab environment setup.
# If running locally, ensure the dataset is downloaded manually.

# # --- COLAB SETUP CODE (Commented out for local .py file) ---
# import zipfile
# from google.colab import files
# print("Step 1: Upload kaggle.json...")
# uploaded = files.upload()
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# with zipfile.ZipFile("chest-xray-pneumonia.zip", 'r') as zip_ref:
#     zip_ref.extractall(".")

# ==========================================
# 2. CONFIGURATION
# ==========================================
# For your local repo, point this to where the data WOULD be.
# If someone clones this, they will change this path to their own data folder.
DATASET_PATH = './chest_xray/chest_xray' 

TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
TEST_DIR = os.path.join(DATASET_PATH, 'test')
VAL_DIR = os.path.join(DATASET_PATH, 'val')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 

# ==========================================
# 3. DATA PROCESSING (VGG16 Specific)
# ==========================================
from tensorflow.keras.applications.vgg16 import preprocess_input

# Check if data exists before running generators (Prevents crash if data is missing)
if os.path.exists(TRAIN_DIR):
    print("Setting up Data Generators...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
    )
    val_generator = test_val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
    )
    test_generator = test_val_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=1, class_mode='binary', shuffle=False
    )
else:
    print(f"Dataset not found at {DATASET_PATH}. Skipping data generation.")

# ==========================================
# 4. BUILD VGG16 MODEL
# ==========================================
print("Building VGG16 Model...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

# ==========================================
# 5. TRAIN (Conditional)
# ==========================================
if os.path.exists(TRAIN_DIR):
    print("Starting Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # ==========================================
    # 6. RESULTS
    # ==========================================
    print("\n--- RESEARCH RESULTS ---")
    predictions = model.predict(test_generator)
    predicted_classes = (predictions > 0.5).astype("int32")
    true_classes = test_generator.classes

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=['Normal', 'Pneumonia']))

    model.save('vgg16_pneumonia_research.h5')
    print("Model saved.")