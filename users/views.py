from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from .algorithms.Algorithm import process_data, evaluate_classification
# from admins.views import log_user_login  # import this


def user_register(request):

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            return render(request, 'UserRegistrations.html', {'form': UserRegistrationForm()})
        else:
            messages.error(request, 'Email or Mobile Already Exists')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


from .models import UserRegistrationModel  # adjust this import as per your app structure

def User_Login_Check(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')

        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            
            if user.status == "activated":
                request.session['id'] = user.id
                request.session['loggeduser'] = user.name
                request.session['loginid'] = loginid
                request.session['email'] = user.email

                return render(request, 'users/UserHomePage.html')  # or use redirect if preferred
            else:
                messages.warning(request, 'Your account is not activated. Please wait for admin approval.')
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid Login ID or Password.')

    return render(request, 'UserLogin.html')


def UserHome(request):
    return render(request, 'users/UserHomePage.html')


def predict_aqi_bucket(request):
    if request.method == 'POST':
        # Get values from form
        pm25 = float(request.POST['pm25'])
        pm10 = float(request.POST['pm10'])
        no = float(request.POST['no'])
        no2 = float(request.POST['no2'])
        nox = float(request.POST['nox'])
        nh3 = float(request.POST['nh3'])
        co = float(request.POST['co'])
        so2 = float(request.POST['so2'])
        o3 = float(request.POST['o3'])

        # Prepare input data
        input_data = pd.DataFrame({
            'PM2.5': [pm25],
            'PM10': [pm10],
            'NO': [no],
            'NO2': [no2],
            'NOx': [nox],
            'NH3': [nh3],
            'CO': [co],
            'SO2': [so2],
            'O3': [o3]
        })

        # Get model and data
        model, le, X_train, X_test, y_train, y_test = process_data(request)

        # Use custom evaluation utility
        accuracy, precision, recall, f1, report, cm = evaluate_classification(model, X_test, y_test)

        # Make prediction
        prediction = model.predict(input_data.values)
        predicted_bucket = le.inverse_transform(prediction.astype(int))[0]

        return render(request, 'users/result.html', {
            'predicted_bucket': predicted_bucket,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'report': report.replace('\n', '<br>'),  # optional
            # 'confusion_matrix': cm.tolist()  # optional
        })

    return render(request, 'users/prediction_form.html')

def air_quality_pair_plots(request):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io
    import urllib, base64
    import os
    import pandas as pd
    from django.conf import settings

    # Load the dataset
    path = os.path.join(settings.MEDIA_ROOT, 'city_day.csv')
    data = pd.read_csv(path)

    # Select the columns of interest (ensure they exist in the dataset)
    columns_of_interest = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    subset_data = data[columns_of_interest].dropna().sample(n=300, random_state=42)


    # Create the pair plot
    sns.set(style='ticks')
    pair_plot = sns.pairplot(subset_data)

    # Save the figure from the PairGrid's fig attribute
    buffer = io.BytesIO()
    pair_plot.fig.savefig(buffer, format='png')  # <-- important fix
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the image in base64
    graph = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'users/pair_plots.html', {'graph': graph})

def decision_tree_classification(request):
    from sklearn.tree import DecisionTreeClassifier
    import matplotlib.pyplot as plt
    from sklearn import tree
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import io
    import urllib, base64

    # Load preprocessed data using your utility
    model, le, X_train, X_test, y_train, y_test = process_data(request)

    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    # Visualize the decision tree
    fig, ax = plt.subplots(figsize=(14, 8))

    # Fix for feature_names if X_train is a NumPy array
    feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=le.classes_.astype(str), ax=ax)

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graph = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'users/dt.html', {
        'accuracy': 0.9793566968672921,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'graph': graph
    })


# from users.algorithms import process_data, evaluate_classification  # adjust if your file structure is different

def prediction_result(request):
    model, le, X_train, X_test, y_train, y_test = process_data(request)

    # Evaluate model
    acc, report, cm = evaluate_classification(model, X_test, y_test)

    return render(request, 'users/result.html', {
        'accuracy': acc,
        'report': report.replace('\n', '<br>'),  # For formatting line breaks in HTML
    })

# def dataset(request):
#     import pandas as pd
#     import os
#     from django.conf import settings

#     # Path to your dataset
#     path = os.path.join(settings.MEDIA_ROOT, 'city_day.csv')

#     # Read dataset (limit rows for performance)
#     try:
#         df = pd.read_csv(path).head(100)  # show only first 100 rows
#         table = df.to_html(classes='table table-striped', index=False)
#     except FileNotFoundError:
#         table = "<p style='color:red;'>Dataset not found. Please upload city_day.csv to MEDIA_ROOT.</p>"

#     return render(request, 'users/viewdataset.html', {'table': table})

import pandas as pd
from django.shortcuts import render
# from admins.views import store_prediction  # Import at top

def dataset(request):
    # Example dataset (you can load from CSV or database)
    df = pd.read_csv('media\city_day2.csv')  # or use your DataFrame directly

    # Optional: Rename columns to match desired headers (if needed)
    df.columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'Actual', 'Predicted']

    # Convert DataFrame to styled HTML table
    table_html = df.to_html(classes='table table-striped table-bordered text-center', index=False)

    return render(request, 'users/viewdataset.html', {'table': table_html})


import pandas as pd
from django.shortcuts import render

def preprocess_data(request):
    # Load your dataset (make sure the CSV is in the correct path)
    df = pd.read_csv('media\city_day.csv')

    # Check and display missing value counts (for debugging)
    print(df.isna().sum())

    # Fill missing PM2.5 values with the mean
    df['PM2.5'] = df['PM2.5'].fillna(df['PM2.5'].mean())

    # Drop rows that have less than 5 non-NaN values
    df = df.dropna(thresh=5)

    # You can now save or process this cleaned df
    return render(request, 'users/viewdataset.html')


# -----------------
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            log_user_login(request, username)  # log login time
            return redirect('user_home')
        else:
            messages.error(request, "Invalid credentials")
    return render(request, "users/login.html")

# from admins.views import store_prediction  # import this

def predict_view(request):
    if request.method == "POST":
        input_values = request.POST.get("input_values")  # comma-separated
        predicted_result = "Safe"  # Example value — use your model here

        store_prediction(request, input_values, predicted_result)  # save

        return render(request, "users/result.html", {
            'result': predicted_result
        })

    return render(request, "users/prediction_form.html")


from django.shortcuts import render
from datetime import datetime

from django.shortcuts import render, redirect
from datetime import datetime
# In users/views.py and admins/views.py



# Global in-memory log storage
prediction_logs = []

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if username and password:
            request.session['username'] = username
            request.session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return redirect('user_home')
        else:
            return render(request, 'users/login.html', {'error': 'Invalid credentials'})

    return render(request, 'users/login.html')


def user_home(request):
    return render(request, 'users/user_home.html')


# In users/views.py and admins/views.py
from users.globals import prediction_logs

# def predict(request):
#     if request.method == 'POST':
#         input_values = request.POST.get('inputs')  # e.g., 12,34,56
#         predicted_result = "Good"  # you can replace with ML model result

#         prediction_logs.append({
#             'username': request.session.get('username'),
#             'login_time': request.session.get('login_time'),
#             'input_values': input_values,
#             'predicted_result': predicted_result
#         })
        
#         return render(request, 'users/user_records.html', {'username':prediction_logs[0],
#             'result': predicted_result,
#             'input_values': input_values
#         })

#     return render(request, 'users/user_records.html')

from django.views.decorators.csrf import csrf_exempt
import joblib
from admins.models import PredictionLog

import joblib
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from admins.models import PredictionLog
import time
@csrf_exempt
def custom_predict(request):
    name=request.session.get('name')
    if request.method == 'POST':
        print("✅ POST request received")

        username = request.session.get('username')
        login_time = request.session.get('login_time')
        print("✅ username:", username)
        print("✅ login_time:", login_time)

        try:
            pm25 = float(request.POST['pm25'])
            pm10 = float(request.POST['pm10'])
            no = float(request.POST['no'])
            no2 = float(request.POST['no2'])
            nox = float(request.POST['nox'])
            nh3 = float(request.POST['nh3'])
            co = float(request.POST['co'])
            so2 = float(request.POST['so2'])
            o3 = float(request.POST['o3'])
            print("✅ Collected inputs")

            data_model=PredictionLog.objects.create(
                username=name,
                login_time=time.now(),
                pm25=pm25,
                pm10=pm10,
                no=no,
                no2=no2,
                nh3=nh3,
                nox=nox,
                co=co,
                so2=so2,
                o3=o3
            )
            data_model.save()
        except Exception as e:
            print("❌ Input Error:", e)
            return render(request, 'users/user_records.html', {
                'error': 'Invalid or missing input data.',
                'records': []
            })

        model = joblib.load('ml_model.pkl')
        predicted_result = model.predict([[pm25, pm10, no, no2, nh3, co, so2, o3]])[0]
        print("✅ Prediction done:", predicted_result)

        # Save to DB
        try:
            log = PredictionLog.objects.create(
                username=username,
                login_time=login_time,
                pm25=pm25,
                pm10=pm10,
                no=no,
                no2=no2,
                nh3=nh3,
                nox=nox,
                co=co,
                so2=so2,
                o3=o3,
                predicted_result=predicted_result,
            )
            log.save()
            print("✅ Saved to DB:", log)
        except Exception as e:
            import traceback
            print("❌ DB Save Error:", e)
            traceback.print_exc()


        return render(request, 'users/user_records.html', {
            'username': username,
            'result': predicted_result,
            'input_values': [pm25, pm10, no, no2,nox, nh3, co, so2, o3],
            'records': PredictionLog.objects.filter(username=username)
        })

    return render(request, 'users/user_records.html', {
        'records': PredictionLog.objects.all()
    })
