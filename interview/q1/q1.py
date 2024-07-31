from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
data = data / 255

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in tqdm(X_test):
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)
    return predictions

def evaluate(y_test, y_pred):
    accuracy = np.mean(y_test == y_pred)
    return accuracy
k=3
y_pred = predict(X_train, y_train, X_test, k)

accuracy = evaluate(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

def save_images(X_test, y_test, y_pred):
    path_save = os.path.join("q1","result")
    os.makedirs(path_save, exist_ok=True)
    
    for i in range(len(X_test)):
        plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
        plt.title(f'True: {y_test[i]}, Pred: {y_pred[i]}')
        plt.axis('off')
        plt.savefig(f'{path_save}/image_{i}.png')
        plt.close()

save_images(X_test, y_test, y_pred)