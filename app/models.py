from django.db import models
from django.contrib.auth.models import User

class Patient(models.Model):
    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=15, unique=True)
    gender = models.CharField(max_length=10, choices=[("Male", "Male"), ("Female", "Female"), ("Other", "Other")])
    age = models.IntegerField()

    def __str__(self):
        return self.name

class Symptoms(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # allow anonymous
    # Features used by the trained numeric model
    Pregnancies = models.IntegerField(null=True, blank=True)
    Glucose = models.FloatField(null=True, blank=True)
    BloodPressure = models.FloatField(null=True, blank=True)
    SkinThickness = models.FloatField(null=True, blank=True)
    Insulin = models.FloatField(null=True, blank=True)
    BMI = models.FloatField(null=True, blank=True)
    DiabetesPedigreeFunction = models.FloatField(null=True, blank=True)
    Age = models.IntegerField()
    # Legacy categorical fields (kept for backward compatibility)
    Gender = models.CharField(max_length=10, null=True, blank=True)
    Polyuria = models.CharField(max_length=10, null=True, blank=True)
    Polydipsia = models.CharField(max_length=10, null=True, blank=True)
    sudden_weight_loss = models.CharField(max_length=10, null=True, blank=True)
    weakness = models.CharField(max_length=10, null=True, blank=True)
    Polyphagia = models.CharField(max_length=10, null=True, blank=True)
    Genital_thrush = models.CharField(max_length=10, null=True, blank=True)
    visual_blurring = models.CharField(max_length=10, null=True, blank=True)
    Itching = models.CharField(max_length=10, null=True, blank=True)
    Irritability = models.CharField(max_length=10, null=True, blank=True)
    delayed_healing = models.CharField(max_length=10, null=True, blank=True)
    partial_paresis = models.CharField(max_length=10, null=True, blank=True)
    muscle_stiffness = models.CharField(max_length=10, null=True, blank=True)
    Alopecia = models.CharField(max_length=10, null=True, blank=True)
    Obesity = models.CharField(max_length=10, null=True, blank=True)
    prediction = models.CharField(max_length=32, null=True, blank=True)   # ADD
    risk_level = models.CharField(max_length=16, null=True, blank=True)   # ADD
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username if self.user else 'anonymous'} - {self.created_at}"
