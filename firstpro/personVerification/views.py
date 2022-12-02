from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import os


from personFunction import process_data  



def index(request):
     template = loader.get_template('indexp.html')
     return HttpResponse(template.render())

def output(request):
#     x = request.POST['cnic']
#     y = request.POST['photo']
#     print("nnnnnnnnnnnn",x,y)

    dcs, perc = process_data(x,y)
    template = loader.get_template('output.html')
#     context = {
#     'dcs': dcs,
#     'perc':perc
#     }
    return HttpResponse(template.render())