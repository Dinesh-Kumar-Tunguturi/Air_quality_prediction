from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import PredictionLog

from django.contrib import admin
from .models import PredictionLog

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ['username', 'login_time', 'predicted_result']
    search_fields = ['username']

