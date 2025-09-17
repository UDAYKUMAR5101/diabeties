from django.db import models

class Patient(models.Model):
    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=15, unique=True)
    gender = models.CharField(max_length=10, choices=[("Male", "Male"), ("Female", "Female"), ("Other", "Other")])
    age = models.IntegerField()

    def __str__(self):
        return self.name
from django.db import models
from django.contrib.auth.models import User

class Symptoms(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
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
    created_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=20, null=True, blank=True)
    risk_level = models.CharField(max_length=10, null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.created_at}"
