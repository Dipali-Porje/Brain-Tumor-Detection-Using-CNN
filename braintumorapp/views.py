

# Create your views here.
from django.shortcuts import render
from .forms import MRIImageForm
from .models import MRIImage
from .model_loader import predict_tumor

def upload_image(request):
    if request.method == "POST":
        form = MRIImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()  # Save uploaded image
            image_path = image_instance.image.path
            prediction = predict_tumor(image_path)  # Get prediction
           
            return render(request, "result.html", {"image": image_instance, "prediction": prediction})
    else:
        form = MRIImageForm()
    return render(request, "upload.html", {"form": form})

