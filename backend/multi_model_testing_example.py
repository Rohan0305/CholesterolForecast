import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.stats

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

try:
    # Import data
    data = pd.read_csv('Heart_health.csv')  
    print("Data loaded successfully.")
    # Format data
    data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure(mmHg)'].str.split('/', expand=True)
    data['Systolic_BP'] = data['Systolic_BP'].astype(float)
    data['Diastolic_BP'] = data['Diastolic_BP'].astype(float)
    data['Smoker'] = data['Smoker'].map({'Yes': 1, 'No': 0})
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

    data_numeric = data.drop(columns=['ID', 'Name', 'Heart Attack'])

    features = ['Age', 'Height(cm)', 'Weight(kg)', 'Systolic_BP', 'Diastolic_BP', 'Glucose(mg/dL)', 'Exercise(hours/week)', 'Smoker', 'Gender']
    X = data_numeric[features]
    y = data_numeric['Cholesterol(mg/dL)']

    # Initialize training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale and standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models with respective random states and max_iter vals
    # KNN and Gaussian NB do not have random states
    models_test = []
    models_test.append(("Logistic Regression", LogisticRegression(random_state=42, max_iter=2000)))
    models_test.append(("SVC", SVC(random_state=42, max_iter=2000)))
    models_test.append(("Linear SVC", LinearSVC(random_state=42, max_iter=2000, dual=True)))
    models_test.append(("K Nearest Neighbors", KNeighborsClassifier()))
    models_test.append(("Decision Tree", DecisionTreeClassifier(random_state=42)))
    models_test.append(("Random Forest", RandomForestClassifier(random_state=42)))
    models_test.append(("Gaussian Naive Bayes", GaussianNB()))
    models_test.append(("MLP Classifier", MLPClassifier(random_state=42, max_iter=2000)))

    # Iterate through all models and score
    models_names = []
    scores = []
    for name, model in models_test:
        model.fit(X_train_scaled, y_train)
        result = model.score(X_test_scaled, y_test)
        models_names.append(name)
        scores.append(result)

    # With the results stored in the data arrays, they can be passed elsewhere
    # Iterate for now and print
    for i in range(len(models_names)):
        print("%s has %0.4f accuracy." % (models_names[i], scores[i]))

except FileNotFoundError:
    print("Error: File not found. Please check the file path and try again.")

except Exception as e:
    print("An error occurred:", e)