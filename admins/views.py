from django.shortcuts import render, redirect
from django.contrib import messages
from users.forms import UserRegistrationForm
from users.models import UserRegistrationModel
from users.globals import prediction_logs  # <- important

# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is =", usrid)

        if usrid == 'ADMIN' and pswd == 'ADMIN':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.error(request, '❌ Invalid credentials. Please try again.')
            return redirect('AdminLogin')  # make sure this name exists in urls.py

    return render(request, 'AdminLogin.html')

def AdminHome(request):
    return render(request, 'admins/AdminHome.html',{})

def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/viewregisterusers.html',{'data':data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/viewregisterusers.html',{'data':data})
    


# def user_records(request):
#     print("LOG COUNT:", len(prediction_logs))
#     print("LOG DATA:", prediction_logs)

#     return render(request, 'admins/user_records.html', {
#         'records': prediction_logs

#     })


def user_records(request):
    print("LOG COUNT:", len(prediction_logs))
    return render(request, 'admins/user_records.html', {
        'records': prediction_logs
    })


from django.shortcuts import render
from admins.models import PredictionLog


def user_records(request):
    records = PredictionLog.objects.all().order_by('-login_time')
    print("✅ Total records fetched:", records.count())
    return render(request, 'admins/user_records.html', {'records': records})
