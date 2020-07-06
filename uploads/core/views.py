from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
from uploads.core.yolo_detection import detection


def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'core/simple_upload.html')


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            conf = "./yolov3-gurmina-brand.cfg"
            weights = "./yolov3-gurmina-brand_last.weights"
            names = "./gurmina-brand.names"
#            conf = "./yolov3-gurmina.cfg"
#            weights = "./yolov3-gurmina_best.weights"
#            names = "./gurmina.names"
            print(form.cleaned_data['detection'])
            print(form.cleaned_data['nms_threshold'])
            print(form.cleaned_data['conf_threshold'])
            print(form.cleaned_data['document'])
            obj = form.save()
            print(obj.document)
            a=detection("media/"+str(obj.document), conf, weights, names, float(form.cleaned_data['detection']), float(form.cleaned_data['nms_threshold']), float(form.cleaned_data['conf_threshold']))
            return render(request, 'core/photo.html', {
            'uploaded_file_url': obj.document.url, 'uploaded_file_name': obj.document.name, 
        })
    else:
        form = DocumentForm()
        form.fields['detection'].initial = 0.2
        form.fields['nms_threshold'].initial = 0.1
        form.fields['conf_threshold'].initial = 0.2
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
