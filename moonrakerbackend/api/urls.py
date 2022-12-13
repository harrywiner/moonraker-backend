from django.urls import path
from . import views

urlpatterns = [
    path('helloworld', views.getData),
    path('anomalies', views.findAnomalies)
]