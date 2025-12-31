
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .utils import predict_toxicity
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save() # This saves the user to the database
            login(request, user)
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, "userapp/register.html", {"form": form})

def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, "userapp/login.html", {"form": form})



def logout_view(request):
    logout(request)
    return redirect("login")

import time

@login_required(login_url="login")
def dashboard(request):
    results = None 
    user_comment = ""
    processing_time = None

    if request.method == "POST":
        user_comment = request.POST.get("text_input", "").strip() 
        
        if user_comment:
            # 1. Basic length validation for BERT (MAX 512 tokens)
            if len(user_comment) > 1000:
                return render(request, "userapp/dashboard.html", {
                    "error": "Input is too long. Please keep it under 1000 characters.",
                    "user_input": user_comment
                })

            try:
                start_time = time.time()
                
                # 2. Call the Ensemble Logic
                results = predict_toxicity(user_comment)
                
                
                processing_time = round(time.time() - start_time, 2)
                
            except Exception as e:
                print(f"Prediction Error: {e}")
                results = {'error': "The AI engine is currently warming up. Please try again in a moment."}

    return render(request, "userapp/dashboard.html", {
        "results": results,
        "user_input": user_comment,
        "processing_time": processing_time
    })