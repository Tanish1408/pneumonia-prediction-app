from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
from PIL import Image
import os

# We define the model variable globally but set it to None initially
model = None

def home(request):
    global model  # Access the global variable
    
    result = None
    confidence = None
    file_url = None

    if request.method == 'POST' and request.FILES['image']:
        # 1. Save the file temporarily
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)
        file_path = fs.path(filename)

        # 2. LAZY LOAD: Only import TensorFlow when we actually need it!
        # This prevents the server from crashing on startup.
        if model is None:
            from tensorflow.keras.models import load_model
            model_path = os.path.join(os.path.dirname(__file__), '..', 'pneumonia_model.h5')
            model = load_model(model_path)

        # 3. Preprocess the image
        img = Image.open(file_path).convert('L')  # Grayscale
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 1)

        # 4. Make Prediction
        prediction = model.predict(img_array)
        
        # 5. Interpret Result
        if prediction[0][0] > 0.5:
            result = "Pneumonia Detected"
            confidence = round(prediction[0][0] * 100, 2)
        else:
            result = "Normal"
            confidence = round((1 - prediction[0][0]) * 100, 2)

    return render(request, 'diagnosis/home.html', {
        'result': result, 
        'confidence': confidence, 
        'file_url': file_url
    })