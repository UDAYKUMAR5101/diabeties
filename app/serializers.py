from rest_framework import serializers
from .models import Patient

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ["id", "name", "phone", "gender", "age"]


from rest_framework import serializers
from .models import Symptoms

class SymptomsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Symptoms
        fields = '__all__'
        read_only_fields = ['user', 'created_at', 'prediction', 'risk_level']

