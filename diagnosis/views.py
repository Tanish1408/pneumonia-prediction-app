from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import numpy as np
from PIL import Image
import os
import tensorflow as tf

# Global variable for the Lite Interpreter
interpreter = None

def load_lite_model():
    """
    Loads the optimized TFLite model from the root or models folder.
    """
    global interpreter
    
    # 1. Look for the file in the project root first
    model_path = os.path.join(settings.BASE_DIR, 'pneumonia_model.tflite')
    
    # 2. If not found, look in the 'models' folder
    if not os.path.exists(model_path):
        model_path = os.path.join(settings.BASE_DIR, 'models', 'pneumonia_model.tflite')

    # 3. Crash if still not found
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find pneumonia_model.tflite! Checked root and 'models/' folder.")
        
    # 4. Load the TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("DEBUG: TFLite Model Loaded Successfully!")

def home(request):
    global interpreter
    
    result = None
    confidence = None
    file_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)
        file_path = fs.path(filename)

        try:
            # --- 1. LOAD MODEL (Lazy Loading) ---
            if interpreter is None:
                load_lite_model()

            # --- 2. GET MODEL DETAILS ---
            # This asks the model: "What image size do you want?"
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Extract expected height/width (e.g., 64 or 150)
            input_shape = input_details[0]['shape'] 
            target_height = input_shape[1]
            target_width = input_shape[2]

            # --- 3. PREPROCESS IMAGE ---
            img = Image.open(file_path).convert('RGB')
            img = img.resize((target_width, target_height))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Reshape to (1, H, W, 3)
            img_array = img_array.reshape(1, target_width, target_height, 3)

            # --- 4. PREDICT (TFLite Style) ---
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            # --- 5. INTERPRET RESULT ---
            pred_value = prediction[0][0]
            
            if pred_value > 0.5:
                result = "Pneumonia Detected"
                confidence = round(pred_value * 100, 2)
            else:
                result = "Normal"
                confidence = round((1 - pred_value) * 100, 2)
                
        except Exception as e:
            print(f"ERROR: {e}")
            result = f"Error: {e}"

    return render(request, 'home.html', {
        'result': result, 
        'confidence': confidence, 
        'file_url': file_url
    })