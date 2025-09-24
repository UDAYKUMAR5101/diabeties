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

    def validate(self, attrs):
        # Ensure at least numeric model features are present
        required_numeric = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        missing = [f for f in required_numeric if attrs.get(f) in (None, "")]
        if missing:
            raise serializers.ValidationError({
                'missing_fields': f"Required numeric features missing: {', '.join(missing)}"
            })
        return attrs