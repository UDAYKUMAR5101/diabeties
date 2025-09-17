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
    Age = models.IntegerField()
    Gender = models.CharField(max_length=10)
    Polyuria = models.CharField(max_length=10)
    Polydipsia = models.CharField(max_length=10)
    sudden_weight_loss = models.CharField(max_length=10)
    weakness = models.CharField(max_length=10)
    Polyphagia = models.CharField(max_length=10)
    Genital_thrush = models.CharField(max_length=10)
    visual_blurring = models.CharField(max_length=10)
    Itching = models.CharField(max_length=10)
    Irritability = models.CharField(max_length=10)
    delayed_healing = models.CharField(max_length=10)
    partial_paresis = models.CharField(max_length=10)
    muscle_stiffness = models.CharField(max_length=10)
    Alopecia = models.CharField(max_length=10)
    Obesity = models.CharField(max_length=10)
    prediction = models.CharField(max_length=32, null=True, blank=True)   # ADD
    risk_level = models.CharField(max_length=16, null=True, blank=True)   # ADD
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username if self.user else 'anonymous'} - {self.created_at}"
