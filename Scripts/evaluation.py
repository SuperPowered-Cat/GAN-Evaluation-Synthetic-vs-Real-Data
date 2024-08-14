import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_and_evaluate(X_train, y_train, X_test, y_test, train_sizes):
    accuracies = []
    for size in train_sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=size, stratify=y_train, random_state=42)
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train_subset, y_train_subset)
        y_pred = svm.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return accuracies

def plot_results(train_sizes, real_accuracies, synthetic_accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, real_accuracies, marker='o', label='Real Data')
    plt.plot(train_sizes, synthetic_accuracies, marker='s', label='Synthetic Data')
    plt.xlabel('Fraction of Training Data Used')
    plt.ylabel('Accuracy')
    plt.title('SVM Performance: Real vs Synthetic Data')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_real_vs_synthetic(X_train_scaled, y_train, X_test_scaled, y_test, synthetic_data):
    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    real_accuracies = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, train_sizes)
    synthetic_accuracies = train_and_evaluate(X_train_scaled, y_train, synthetic_data, y_train, train_sizes)
    plot_results(train_sizes, real_accuracies, synthetic_accuracies)

if __name__ == "__main__":
    X_train_scaled = np.load('X_train_scaled.npy')
    X_test_scaled = np.load('X_test_scaled.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    synthetic_data = np.load('synthetic_data.npy')  # Assuming you have this already
    evaluate_real_vs_synthetic(X_train_scaled, y_train, X_test_scaled, y_test, synthetic_data)
