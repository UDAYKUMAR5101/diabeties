from django.urls import path
from .views import SymptomsView, create_patient, get_patient_details

urlpatterns = [
    path("create/", create_patient, name="create_patient"),
    path("get-patient/", get_patient_details, name="get_patient_details"),
    path('predict/', SymptomsView.as_view(), name='predict'),
]
