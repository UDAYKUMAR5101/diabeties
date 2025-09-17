from rest_framework import serializers
from .models import Patient, Symptoms

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ["id", "name", "phone", "gender", "age"]

class SymptomsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Symptoms
        fields = '__all__'
        read_only_fields = ['user', 'created_at', 'prediction', 'risk_level']