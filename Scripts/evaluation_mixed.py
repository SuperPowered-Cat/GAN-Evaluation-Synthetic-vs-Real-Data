import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_and_evaluate_mixed(X_train, y_train, X_test, y_test, synthetic_X, synthetic_y, mix_ratios):
    accuracies_real = []
    accuracies_mixed = []
    
    for ratio in mix_ratios:
        svm_real = SVC(kernel='rbf', random_state=42)
        svm_real.fit(X_train, y_train)
        y_pred_real = svm_real.predict(X_test)
        accuracies_real.append(accuracy_score(y_test, y_pred_real))
        
        n_synthetic = int(len(X_train) * ratio)
        X_mixed = np.vstack((X_train, synthetic_X[:n_synthetic]))
        y_mixed = np.concatenate((y_train, synthetic_y[:n_synthetic]))
        
        svm_mixed = SVC(kernel='rbf', random_state=42)
        svm_mixed.fit(X_mixed, y_mixed)
        y_pred_mixed = svm_mixed.predict(X_test)
        accuracies_mixed.append(accuracy_score(y_test, y_pred_mixed))
    
    return accuracies_real, accuracies_mixed

def plot_mixed_results(mix_ratios, accuracies_real, accuracies_mixed):
    plt.figure(figsize=(10, 6))
    plt.plot(mix_ratios, accuracies_real, marker='o', label='Real Data Only')
    plt.plot(mix_ratios, accuracies_mixed, marker='s', label='Mixed Real and Synthetic Data')
    plt.xlabel('Ratio of Synthetic to Real Data')
    plt.ylabel('Accuracy')
    plt.title('SVM Performance: Real vs Mixed Data')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X_train_scaled = np.load('X_train_scaled.npy')
    X_test_scaled = np.load('X_test_scaled.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    synthetic_data = np.load('synthetic_data.npy')  # Assuming you have this already
    synthetic_y = np.random.choice([0, 1], size=len(synthetic_data))  # Replace with actual labels if available
    
    mix_ratios = [0, 0.25, 0.5, 0.75, 1.0]
    accuracies_real, accuracies_mixed = train_and_evaluate_mixed(X_train_scaled, y_train, X_test_scaled, y_test, synthetic_data, synthetic_y, mix_ratios)
    plot_mixed_results(mix_ratios, accuracies_real, accuracies_mixed)
