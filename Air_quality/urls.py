from django.contrib import admin
from django.urls import path
from users import views as usr
from Air_quality import views
from admins import views as admin_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('UserLogin/', usr.User_Login_Check, name='login'),
    path('UserHome/', usr.UserHome, name='user_home'),
    path('UserRegister/', usr.user_register, name='user_register'),

    path('predict/', usr.predict_aqi_bucket, name='predict_aqi'),
    path('plot/', usr.air_quality_pair_plots, name='air_quality_pair_plots'),
    path('AdminLogin/', admin_views.AdminLoginCheck, name='AdminLogin'),
    path('RegisterUsersView/', admin_views.AdminHome, name='RegisterUsersView'), 
    path('userlist/', admin_views.RegisterUsersView, name='userlist'), 
    path('ActivaUsers/', admin_views.ActivaUsers, name='ActivaUsers'),
    path('decision-tree/', usr.decision_tree_classification, name='decision_tree'),
    path('predict/', usr.predict_aqi_bucket, name='predict'), 
    path('dataset/',usr.dataset,name="dataset"),
    path('user-records/', admin_views.user_records, name='user_records'),
    path('predict1/', usr.predict_view, name='predict1'),


]
