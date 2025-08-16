import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from django.conf import settings
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def process_data(request):
    # Load the data
    path = os.path.join(settings.BASE_DIR, 'media/city_day.csv')
    d = pd.read_csv(path)

    # Fill missing values with column means
    columns_to_fill = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
    for col in columns_to_fill:
        d[col].fillna(d[col].mean(), inplace=True)

    # Drop irrelevant columns
    for col in ['Benzene', 'Toluene', 'Xylene']:
        if col in d.columns:
            d.drop(col, axis=1, inplace=True)

    # Encode target variable (AQI Category)
    le = LabelEncoder()
    d['AQI_Bucket'] = le.fit_transform(d['AQI_Bucket'])

    # Separate features and labels
    X = d.iloc[:, 2:11].values
    y = d['AQI_Bucket'].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=2)

    # Train classification model
    model = RandomForestClassifier()
    # model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    

    return model, le, X_train, X_test, y_train, y_test


# def evaluate_classification(model, X_test, y_test):
#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred, zero_division=1)
#     cm = confusion_matrix(y_test, y_pred)

#     return acc, report, cm


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, prec, rec, f1, report, cm


# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
#     confusion_matrix
# )

# def process_data():
#     # Load CSV file (make sure the file is in the same directory)
#     path = 'media\city_day.csv'
#     df = pd.read_csv(path)

#     # Fill missing values with column mean
#     features = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
#     for col in features:
#         df[col].fillna(df[col].mean(), inplace=True)

#     # Drop irrelevant columns if they exist
#     df.drop(columns=['Benzene', 'Toluene', 'Xylene'], errors='ignore', inplace=True)

#     # Drop rows where AQI_Bucket is missing
#     df.dropna(subset=['AQI_Bucket'], inplace=True)

#     # Encode the AQI_Bucket column
#     le = LabelEncoder()
#     df['AQI_Bucket'] = le.fit_transform(df['AQI_Bucket'])

#     # Separate features and target
#     X = df[features].values
#     y = df['AQI_Bucket'].values

#     # Split into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2)

#     # Train Decision Tree model
#     model = DecisionTreeClassifier()
#     model.fit(X_train, y_train)

#     return model, le, X_test, y_test, features


# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred, average='weighted', zero_division=1)
#     rec = recall_score(y_test, y_pred, average='weighted', zero_division=1)
#     f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
#     report = classification_report(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)

#     print("\n📊 Model Evaluation:")
#     print(f"Accuracy: {acc:.2f}")
#     print(f"Precision: {prec:.2f}")
#     print(f"Recall: {rec:.2f}")
#     print(f"F1 Score: {f1:.2f}")
#     print("\nClassification Report:\n", report)
#     print("Confusion Matrix:\n", cm)


# def manual_prediction(model, le, features):
#     print("\n🔍 Enter pollutant values for AQI prediction:")
#     user_input = []
#     for feature in features:
#         while True:
#             try:
#                 value = float(input(f"Enter {feature}: "))
#                 user_input.append(value)
#                 break
#             except ValueError:
#                 print("Invalid input. Please enter a number.")

#     input_array = np.array(user_input).reshape(1, -1)
#     prediction = model.predict(input_array)

#     print("\nModel Prediction (Encoded):", prediction[0])
#     print("Known AQI Buckets (Encoded):", list(range(len(le.classes_))), "=>", list(le.classes_))

#     try:
#         predicted_class = le.inverse_transform(prediction)[0]
#     except Exception as e:
#         predicted_class = "Unknown (Out of trained label range)"

#     print(f"\n✅ Predicted AQI Category: {predicted_class}")


# if __name__ == "__main__":
#     model, le, X_test, y_test, features = process_data()
#     evaluate_model(model, X_test, y_test)
#     manual_prediction(model, le, features)



# python users/algorithms/Algorithm.py


# data set from kaggle
# https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023 