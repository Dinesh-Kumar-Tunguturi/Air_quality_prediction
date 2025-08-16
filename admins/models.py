from django.db import models

# Create your models here.

from django.db import models

class PredictionLog(models.Model):
    username = models.CharField(max_length=100)
    login_time = models.DateTimeField()
    pm25 = models.FloatField()
    pm10 = models.FloatField()
    no = models.FloatField()
    no2 = models.FloatField()
    nh3 = models.FloatField()
    nox = models.FloatField(null=True)
    co = models.FloatField()
    so2 = models.FloatField()
    o3 = models.FloatField()
    predicted_result = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.username} - {self.predicted_result}"
