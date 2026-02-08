from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model once
model = load_model('models/pneumonia_model.h5')

def home(request):
    context = {'result': None}
    
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)
        
        # Preprocess
        file_path = os.path.join('media', filename)
        img = image.load_img(file_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        prediction = model.predict(img_array)
        
        # Logic to set Result AND Color
        if prediction[0][0] > 0.5:
            result = "PNEUMONIA DETECTED"
            color = "#dc3545"  # Bootstrap Red (Danger)
        else:
            result = "NORMAL"
            color = "#28a745"  # Bootstrap Green (Success)
            
        # PASS THE COLOR TO THE TEMPLATE HERE!
        context = {
            'result': result,
            'color': color,       # <--- THIS IS THE MISSING KEY
            'file_url': file_url,
            'confidence': round(prediction[0][0] * 100, 2)
        }

    return render(request, 'home.html', context)