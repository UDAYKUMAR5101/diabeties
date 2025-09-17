from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Patient
from .serializers import PatientSerializer

@api_view(["POST"])
def create_patient(request):
    serializer = PatientSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.conf import settings
import os
import joblib
import pandas as pd
from .models import Symptoms
from .serializers import SymptomsSerializer

# Get base directory of the app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model and encoders
model_path = os.path.join(BASE_DIR, 'models', 'diabetes_gb_model.pkl')
label_encoders_path = os.path.join(BASE_DIR, 'models', 'label_encoders.pkl')
target_encoder_path = os.path.join(BASE_DIR, 'models', 'target_encoder.pkl')

# Load the model, label encoders, and target encoder
model = joblib.load(model_path)
label_encoders = joblib.load(label_encoders_path)
target_encoder = joblib.load(target_encoder_path)

class SymptomsView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = SymptomsSerializer(data=request.data)
        if serializer.is_valid():
            # Always save record; attach user if authenticated else store as anonymous
            if request.user.is_authenticated:
                instance = serializer.save(user=request.user)
            else:
                instance = serializer.save()
            
            # Prepare input for ML model
            features = [
                'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden_weight_loss', 'weakness',
                'Polyphagia', 'Genital_thrush', 'visual_blurring', 'Itching', 'Irritability',
                'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'Alopecia', 'Obesity'
            ]
            
            input_data = pd.DataFrame([serializer.validated_data])

            # Rename columns to match training feature names (spaces instead of underscores)
            rename_map = {
                'sudden_weight_loss': 'sudden weight loss',
                'Genital_thrush': 'Genital thrush',
                'visual_blurring': 'visual blurring',
                'delayed_healing': 'delayed healing',
                'partial_paresis': 'partial paresis',
                'muscle_stiffness': 'muscle stiffness',
            }
            input_data.rename(columns=rename_map, inplace=True)

            # Align columns order to model expectations if available
            if hasattr(model, 'feature_names_in_'):
                expected_cols = list(model.feature_names_in_)
                input_data = input_data.reindex(columns=expected_cols)
            
            # Apply label encoding to categorical columns
            for col, le in label_encoders.items():
                if col in input_data.columns:
                    input_data[col] = le.transform(input_data[col])
            
            # Predict
            pred_class = model.predict(input_data)[0]
            pred_prob = model.predict_proba(input_data)[0].max()
            risk_level = f"{pred_prob*100:.2f}%"
            predicted_label = target_encoder.inverse_transform([pred_class])[0]

            # Persist prediction on the saved instance
            instance.prediction = predicted_label
            instance.risk_level = risk_level
            instance.save(update_fields=['prediction', 'risk_level'])

            result = {
                "prediction": predicted_label,
                "risk_level": risk_level
            }

            return Response(result, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        # Get all records for user; return empty list if anonymous
        if not request.user.is_authenticated:
            return Response([])
        symptoms = Symptoms.objects.filter(user=request.user).order_by('-created_at')
        serializer = SymptomsSerializer(symptoms, many=True)
        return Response(serializer.data)