# 1. Import Django models and setup
from django.contrib.auth.models import User  # default User model
from admins.models import PredictionLog
from datetime import datetime

#  2. Create a user (if not already exists)
user, created = User.objects.get_or_create(username='dinesh')

#  3. Create PredictionLog entries related to the user
PredictionLog.objects.create(
    username='dinesh',
    login_time=datetime.now(),
    input_values='PM2.5:34, PM10:45, CO:0.7',
    predicted_result='Good'
)
PredictionLog.objects.create(
    username='dinesh',
    login_time=datetime.now(),
    input_values='PM2.5:76, PM10:88, CO:1.1',
    predicted_result='Moderate'
)

#  4. JOIN operation - fetch PredictionLogs with user data
print("\n--- JOIN (Simulated): Logs with usernames ---")
logs = PredictionLog.objects.all()
for log in logs:
    print(f"User: {log.username}, Result: {log.predicted_result}")

#  5. Simulate `pop()` behavior by converting QuerySet to list
print("\n--- POP behavior (from list) ---")
logs_list = list(PredictionLog.objects.all())
if logs_list:
    popped_log = logs_list.pop()  # removes last
    print(f"Popped Record (not deleted from DB): {popped_log.predicted_result}")

#  6. Delete last record from DB
print("\n--- DELETE last record from DB ---")
last_log = PredictionLog.objects.last()
if last_log:
    print("Deleting:", last_log.predicted_result)
    last_log.delete()

#  7. Count remaining records
print("\nTotal remaining PredictionLog records:", PredictionLog.objects.count())


# PS C:\Users\saiku\OneDrive\Desktop\Air_quality prediction\Air_quality> py manage.py shell                                                                        
# Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# (InteractiveConsole)
# >>> # 1. Import Django models and setup
# >>> from django.contrib.auth.models import User  # default User model
# >>> from admins.models import PredictionLog
# >>> from datetime import datetime
# >>>
# >>> #  2. Create a user (if not already exists)
# >>> user, created = User.objects.get_or_create(username='Meena mam')
# >>> 
# >>> #  3. Create PredictionLog entries related to the user
# >>> PredictionLog.objects.create(
# ...     username='dinesh',
# ...     login_time=datetime.now(),
# ...     input_values='PM2.5:34, PM10:45, CO:0.7',
# ...     predicted_result='Good'
# ... )
# Traceback (most recent call last):
#   File "<console>", line 1, in <module>
#   File "C:\Users\saiku\AppData\Local\Programs\Python\Python38\lib\site-packages\django\db\models\manager.py", line 87, in manager_method
#     return getattr(self.get_queryset(), name)(*args, **kwargs)
#   File "C:\Users\saiku\AppData\Local\Programs\Python\Python38\lib\site-packages\django\db\models\query.py", line 656, in create
#     obj = self.model(**kwargs)
#   File "C:\Users\saiku\AppData\Local\Programs\Python\Python38\lib\site-packages\django\db\models\base.py", line 567, in __init__
#     raise TypeError(
# TypeError: PredictionLog() got unexpected keyword arguments: 'input_values'
# >>> PredictionLog.objects.create(
# ...     username='dinesh',
# ...     login_time=datetime.now(),
# ...     input_values='PM2.5:76, PM10:88, CO:1.1',
# ...     predicted_result='Moderate'
# ... )
# Traceback (most recent call last):
#   File "<console>", line 1, in <module>
#   File "C:\Users\saiku\AppData\Local\Programs\Python\Python38\lib\site-packages\django\db\models\manager.py", line 87, in manager_method
#     return getattr(self.get_queryset(), name)(*args, **kwargs)
#   File "C:\Users\saiku\AppData\Local\Programs\Python\Python38\lib\site-packages\django\db\models\query.py", line 656, in create
#     obj = self.model(**kwargs)
#   File "C:\Users\saiku\AppData\Local\Programs\Python\Python38\lib\site-packages\django\db\models\base.py", line 567, in __init__
#     raise TypeError(
# TypeError: PredictionLog() got unexpected keyword arguments: 'input_values'
# >>>
# >>> #  4. JOIN operation - fetch PredictionLogs with user data
# >>> print("\n--- JOIN (Simulated): Logs with usernames ---")

# --- JOIN (Simulated): Logs with usernames ---
# >>> logs = PredictionLog.objects.all()
# >>> for log in logs:
# ...     print(f"User: {log.username}, Result: {log.predicted_result}")
# ...
# User: Dinesh, Result: good
# User: sai, Result: severe
# User: Dinesh, Result: Severe
# User: Srinath, Result: Poor
# User: charan, Result: Moderate
# >>> #  5. Simulate `pop()` behavior by converting QuerySet to list
# >>> print("\n--- POP behavior (from list) ---")

# --- POP behavior (from list) ---
# >>> logs_list = list(PredictionLog.objects.all())
# >>> if logs_list:
# ...     popped_log = logs_list.pop()  # removes last
# ...     print(f"Popped Record (not deleted from DB): {popped_log.predicted_result}")
# ...
# Popped Record (not deleted from DB): Moderate
# >>> #  6. Delete last record from DB
# >>> print("\n--- DELETE last record from DB ---")

# --- DELETE last record from DB ---
# >>> last_log = PredictionLog.objects.last()
# >>> if last_log:
# ...     print("Deleting:", last_log.predicted_result)
# ...     last_log.delete()
# ...
# Deleting: Moderate
# (1, {'admins.PredictionLog': 1})
# >>> #  7. Count remaining records
# >>> print("\nTotal remaining PredictionLog records:", PredictionLog.objects.count())

# Total remaining PredictionLog records: 4
# >>>