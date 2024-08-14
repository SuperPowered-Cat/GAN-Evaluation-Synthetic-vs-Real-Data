import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def encode_target(y):
    le = LabelEncoder()
    return le.fit_transform(y)

def preprocess_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X, y = load_data('Pumpkin_Seeds_Dataset.csv')
    y = encode_target(y)
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(X, y)
