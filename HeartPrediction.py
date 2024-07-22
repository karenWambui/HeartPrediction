import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
Heart_failure = pd.read_csv("Heart_failure.csv")
Heart_failure['HeartDisease'] = Heart_failure['HeartDisease'].astype(str)
columns_to_convert = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
Heart_failure_encoded = pd.get_dummies(Heart_failure, columns=columns_to_convert)

# Split the data into input features (X) and target variable (y)
X = Heart_failure.drop("HeartDisease", axis=1)
y = Heart_failure["HeartDisease"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('skewness_corrector', PowerTransformer(method='yeo-johnson', standardize=True))
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_features),
        ('categorical', categorical_transformer, categorical_features)
    ]
)

# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier()

# Create a pipeline that first transforms the data then fits the model
pipeline_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', knn)
])

# Define hyperparameter grid for KNN
param_grid = {
    'classifier__leaf_size': [1, 2, 3, 4, 5],
    'classifier__n_neighbors': [1, 3, 5, 7, 9],
    'classifier__p': [1, 2]
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline_knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict and evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best parameters: {best_params}')
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
