from django.shortcuts import render, redirect
# from django.urls import reverse
from .models import *
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
from django.shortcuts import render
from django.http import HttpResponse
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf



def index(request):
    return render(request, 'index.html')

def preprocess_image(image_path, target_size=(256, 256)):
    try:
        img = image.load_img(image_path, target_size=target_size)
    except:
        raise ValueError("Invalid image format or path.")
        
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def register(request):
    if request.method == "POST":
        name = request.POST['name']
        email = request.POST['email']
        password = request.POST['password']
        c_password = request.POST['c_password']
        if password == c_password:
            if user.objects.filter(email=email).exists():
                return render(request, 'register.html', {'message': 'User with this email already exists'})
            new_user = user(name=name, email=email, password=password)
            new_user.save()

            return render(request, 'login.html', {'message': 'Successfully Registerd!'})
        return render(request, 'register.html', {'message': 'Password and Conform Password does not match!'})
    return render(request, 'register.html')


def login(request):
    if request.method == "POST":
        email = request.POST['email']
        password1 = request.POST['password']
        
        try:
            user_obj = user.objects.get(email=email)
            print(3333, user_obj)
        except user.DoesNotExist:
            return render(request, 'login.html', {'message': 'nvalid Username or Password!'})
        
        password2 = user_obj.password
        if password1 == password2:
            return redirect('home')
        else:
            return render(request, 'login.html', {'message': 'Invalid Username or Password!'})
    return render(request, 'login.html')

def home(request):
    return render(request, 'home.html')


# Define function to prepare image
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)  # Load the image
    img_array = img_to_array(img)  # Convert the image to numpy array
    img_array = img_array / 255.0  # Scale the image (if your model requires normalization)
    img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension
    return img_array


def model(request):
    if request.method == "POST":
        algorithm = request.POST['algorithm']
        
        if algorithm == "CNN":
            accuracy = 97

        elif algorithm == "Mobile Net":
            accuracy = 90

        else:
            accuracy = 98
        
        return render(request, 'model.html', {'accuracy': accuracy, "algorithm" : algorithm})
    return render(request, 'model.html')




def upload(request):
    if request.method == "POST":
        myfile = request.FILES['image']
        fn = myfile.name
        mypath = os.path.join('static/img', fn)
        with open(mypath, 'wb+') as destination:
            for chunk in myfile.chunks():
                destination.write(chunk)
        
        # Prepare the image
        img_array = preprocess_image(mypath, target_size=(176, 176))
    
        # Load saved models
        models = []
        for i in range(1, 4):
            model_path = f"model/model_0{i}.h5"
            model = tf.keras.models.load_model(model_path)
            models.append(model)
        
        # Make prediction using the ensemble of models
        predictions = [model.predict(img_array) for model in models]
        ensemble_prediction = np.mean(predictions, axis=0)
        predicted_class = np.argmax(ensemble_prediction, axis=1)[0]  # Get the class with highest probability

        class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        predicted_class_label = class_labels[predicted_class]

        precautions = {}

        if predicted_class_label == "MildDemented":
            precautions = {
                "Reasons ": "Age-related cognitive decline, genetic predisposition, brain injury, vascular issues, lifestyle factors, chronic diseases, mental health disorders, medication side effects.",
                "Precautions": "Regular mental stimulation, healthy diet, physical exercise, social interaction, sleep hygiene, monitor health conditions, avoid smoking and limit alcohol, regular check-ups, manage stress, medication review."
            }
        elif predicted_class_label == "ModerateDemented":
            precautions = {
                "Reasons ": "Cognitive decline, neurodegenerative diseases, age-related risk factors, genetic predisposition, lifestyle factors.",
                "Precautions": "Medication adherence, cognitive stimulation, safety measures, supervised care, routine and familiar environment."
            }
        elif predicted_class_label == "NonDemented":
            precautions = {
                "Reasons": "Healthy aging, effective cognitive reserve, good physical health, balanced nutrition, social engagement.",
                "Precautions": "Regular mental stimulation, physical activity, monitor chronic conditions, healthy sleep habits, avoid excessive alcohol and smoking."
            }
        elif predicted_class_label == "VeryMildDemented":
            precautions = {
                "Reasons ": "Age-related cognitive decline, genetic factors, neurological conditions, cardiovascular health, brain injuries or infections.",
                "Precautions": "Regular cognitive exercises, healthy lifestyle, medical monitoring, social engagement, safety measures."
            }

        return render(request, 'upload.html', {'path': mypath, 'prediction': predicted_class_label, "precautions": precautions})

    return render(request, 'upload.html')

