from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import numpy as np
from PIL import Image
import os

# --- HELPER FUNCTION TO FIND THE FILE ---
def find_model_file():
    """
    Searches for 'pneumonia_model.h5' in the project directory
    and subdirectories.
    """
    search_filename = 'pneumonia_model.h5'
    
    # Start searching from the Base Directory
    start_dir = settings.BASE_DIR
    
    print(f"DEBUG: Starting search in: {start_dir}")
    
    # Walk through all folders and subfolders
    for root, dirs, files in os.walk(start_dir):
        if search_filename in files:
            found_path = os.path.join(root, search_filename)
            print(f"DEBUG: FOUND IT! Model is at: {found_path}")
            return found_path
            
    # If not found, try going up one level (just in case)
    parent_dir = os.path.dirname(start_dir)
    for root, dirs, files in os.walk(parent_dir):
        if search_filename in files:
            found_path = os.path.join(root, search_filename)
            print(f"DEBUG: FOUND IT (in parent dir)! Model is at: {found_path}")
            return found_path

    return None

# Global variable
model = None

def home(request):
    global model
    
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
            # --- LAZY LOAD MODEL ---
            if model is None:
                from tensorflow.keras.models import load_model
                
                # Use our finder function
                model_path = find_model_file()
                
                if not model_path:
                    # If we STILL can't find it, list all files to help debug
                    print("DEBUG: Listing all files in BASE_DIR to help you find it:")
                    for root, dirs, files in os.walk(settings.BASE_DIR):
                        for file in files:
                            print(os.path.join(root, file))
                    raise FileNotFoundError("Could not find pneumonia_model.h5 anywhere! Check your terminal logs.")

                model = load_model(model_path)

            # --- PREPROCESS & PREDICT ---
            # --- OLD CODE (Delete this) ---
            # img = Image.open(file_path).convert('L')  <-- 'L' means Grayscale
            # img_array = img_array.reshape(1, 150, 150, 1)

            # --- NEW CODE (Paste this instead) ---
            # 1. Convert to RGB (3 channels)
            img = Image.open(file_path).convert('RGB') 
            img = img.resize((64, 64))
            img_array = np.array(img) / 255.0
            
            # 2. Reshape to (1, 64, 64, 3)
            img_array = img_array.reshape(1, 64, 64, 3)

            prediction = model.predict(img_array)
            
            if prediction[0][0] > 0.5:
                result = "Pneumonia Detected"
                confidence = round(prediction[0][0] * 100, 2)
            else:
                result = "Normal"
                confidence = round((1 - prediction[0][0]) * 100, 2)
                
        except Exception as e:
            print(f"ERROR: {e}")
            result = f"Error: {e}"

    return render(request, 'home.html', {
        'result': result, 
        'confidence': confidence, 
        'file_url': file_url
    })