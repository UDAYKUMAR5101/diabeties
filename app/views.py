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

@api_view(["GET"])
def get_patient_details(request):
    """
    Validate patient login based on name and phone number.
    Expected query parameters: ?name=patient_name&phone=phone_number
    Returns login success/failure message.
    """
    name = request.query_params.get('name')  # Changed from request.data.get('name')
    phone = request.query_params.get('phone')  # Changed from request.data.get('phone')
    
    # Validate required fields
    if not name or not phone:
        return Response(
            {"message": "Both 'name' and 'phone' are required as query parameters"}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        # Check if patient exists with given name and phone
        patient = Patient.objects.get(name=name, phone=phone)
        return Response(
            {"message": "Login successfully"}, 
            status=status.HTTP_200_OK
        )
    
    except Patient.DoesNotExist:
        return Response(
            {"message": "Not valid details"}, 
            status=status.HTTP_401_UNAUTHORIZED
        )
    except Exception as e:
        return Response(
            {"message": f"An error occurred: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
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

# Construct the full path to the model and numeric imputer
model_path = os.path.join(BASE_DIR, 'models', 'diabetes_gb_model.pkl')
num_imputer_path = os.path.join(BASE_DIR, 'models', 'num_imputer.pkl')

# Load the model and numeric imputer
model = joblib.load(model_path)
num_imputer = joblib.load(num_imputer_path)

class SymptomsView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = SymptomsSerializer(data=request.data)
        if serializer.is_valid():
            try:
                # Always save record; attach user if authenticated else store as anonymous
                if request.user.is_authenticated:
                    instance = serializer.save(user=request.user)
                else:
                    instance = serializer.save()
                
                # Prepare input for numeric model features only
                input_data = pd.DataFrame([{
                    'Pregnancies': serializer.validated_data.get('Pregnancies'),
                    'Glucose': serializer.validated_data.get('Glucose'),
                    'BloodPressure': serializer.validated_data.get('BloodPressure'),
                    'SkinThickness': serializer.validated_data.get('SkinThickness'),
                    'Insulin': serializer.validated_data.get('Insulin'),
                    'BMI': serializer.validated_data.get('BMI'),
                    'DiabetesPedigreeFunction': serializer.validated_data.get('DiabetesPedigreeFunction'),
                    'Age': serializer.validated_data.get('Age'),
                }])

                # Ensure column order matches training
                expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                input_data = input_data.reindex(columns=expected_cols)

                # Impute missing numeric values with the same imputer used in training
                input_data_imputed = num_imputer.transform(input_data)
                
                # Predict with error handling
                try:
                    pred_class = model.predict(input_data_imputed)[0]
                    pred_prob = model.predict_proba(input_data_imputed)[0].max()
                    risk_level = f"{pred_prob*100:.2f}%"
                    predicted_label = 'Diabetic' if pred_class == 1 else 'Non-Diabetic'
                except Exception as e:
                    return Response(
                        {"error": f"Prediction failed: {str(e)}"}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Persist prediction on the saved instance
                instance.prediction = predicted_label
                instance.risk_level = risk_level
                instance.save(update_fields=['prediction', 'risk_level'])

                result = {
                    "prediction": predicted_label,
                    "risk_level": risk_level
                }

                return Response(result, status=status.HTTP_201_CREATED)
                
            except Exception as e:
                return Response(
                    {"error": f"Failed to process request: {str(e)}"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        # Get all records for user; return empty list if anonymous
        if not request.user.is_authenticated:
            return Response([])
        symptoms = Symptoms.objects.filter(user=request.user).order_by('-created_at')
        serializer = SymptomsSerializer(symptoms, many=True)
        return Response(serializer.data)