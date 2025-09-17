from django.urls import path
from .views import SymptomsView, create_patient

urlpatterns = [
    path("create/", create_patient, name="create_patient"),
    path('predict/', SymptomsView.as_view(), name='predict'),
]
