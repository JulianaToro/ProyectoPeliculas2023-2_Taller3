from django.urls import path
from . import views

urlpatterns = [
    path('', views.recomendacion, name='recomendacion'),
    
]
