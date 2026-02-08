import tensorflow as tf
import os

# --- 1. FIND THE FILE ---
# We look specifically in the 'models' folder now
model_path = 'models/pneumonia_model.h5'

if not os.path.exists(model_path):
    # Try looking one level up just in case the script is inside 'models' too
    model_path = '../models/pneumonia_model.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Could not find model at {model_path}")
        print("Please check that you are running this script from the root project folder.")
        exit()

print(f"✅ Found model at: {model_path}")

# --- 2. CONVERT TO LITE ---
print("Loading model... (this may take a moment)")
model = tf.keras.models.load_model(model_path)

print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# --- 3. SAVE THE NEW FILE ---
# We will save the new lite model in the ROOT folder (next to manage.py)
output_file = 'pneumonia_model.tflite'

with open(output_file, 'wb') as f:
    f.write(tflite_model)

print(f"🎉 SUCCESS! Created {output_file}")
print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")