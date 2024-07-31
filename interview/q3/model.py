import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pickle
import os

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
data = data / 255

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

def create_triplets(X, y):
    triplets = []
    classes = np.unique(y)
    
    for c in classes:
        same_class = np.where(y == c)[0]
        diff_class = np.where(y != c)[0]
        
        for i in range(len(same_class)):
            anchor = X[same_class[i]]
            positive = X[np.random.choice(same_class)]
            negative = X[np.random.choice(diff_class)]
            
            triplets.append([anchor, positive, negative])
    
    return np.array(triplets)


train_triplets = create_triplets(X_train, y_train)

class DeepLearningModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for embedding
        return self.a2

    def triplet_loss(self, anchor, positive, negative, margin=0.7):
        dist_pos = np.sum(np.square(anchor - positive), axis=1)
        dist_neg = np.sum(np.square(anchor - negative), axis=1)
        loss = np.maximum(0, dist_pos - dist_neg + margin)
        return np.mean(loss)

    def backward(self, anchor, positive, negative, learning_rate=0.001):
        batch_size = anchor.shape[0]

        anchor_out = self.forward(anchor)
        positive_out = self.forward(positive)
        negative_out = self.forward(negative)

        # Compute gradient 
        da_anchor = 2 * (anchor_out - positive_out) - 2 * (anchor_out - negative_out)
        da_positive = -2 * (anchor_out - positive_out)
        da_negative = 2 * (anchor_out - negative_out)

        #cache
        dW1, db1, dW2, db2 = 0, 0, 0, 0

        # Backpropagation for anchor
        self.forward(anchor)  # Load a1, z1
        dz2 = da_anchor
        dW2 += np.dot(self.a1.T, dz2)
        db2 += np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU gradient
        dW1 += np.dot(anchor.T, dz1)
        db1 += np.sum(dz1, axis=0, keepdims=True)

        # Backpropagation for positive
        self.forward(positive)  # Load a1, z1
        dz2 = da_positive
        dW2 += np.dot(self.a1.T, dz2)
        db2 += np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 += np.dot(positive.T, dz1)
        db1 += np.sum(dz1, axis=0, keepdims=True)

        # Backpropagation for negative
        self.forward(negative)  # Load a1, z1
        dz2 = da_negative
        dW2 += np.dot(self.a1.T, dz2)
        db2 += np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 += np.dot(negative.T, dz1)
        db1 += np.sum(dz1, axis=0, keepdims=True)

        # Get avg gradients
        dW1 /= (3 * batch_size)
        db1 /= (3 * batch_size)
        dW2 /= (3 * batch_size)
        db2 /= (3 * batch_size)

        # Update w and b
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, triplets, epochs=10, batch_size=32):
        for epoch in range(epochs):
            np.random.shuffle(triplets)
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i+batch_size]
                anchors = batch[:, 0]
                positives = batch[:, 1]
                negatives = batch[:, 2]

                anchor_embeddings = self.forward(anchors)
                positive_embeddings = self.forward(positives)
                negative_embeddings = self.forward(negatives)

                loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                self.backward(anchors, positives, negatives)

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss}")

    def evaluate(self, X_test, y_test):
        embeddings = self.forward(X_test)
        correct = 0
        total = len(y_test)

        for i in range(total):
            distances = np.sum(np.square(embeddings - embeddings[i]), axis=1)
            nearest_neighbor = np.argmin(distances[np.arange(len(distances)) != i])
            if y_test[nearest_neighbor] == y_test[i]:
                correct += 1

        accuracy = correct / total
        return accuracy

    def predict(self, X_image, X_train, y_train):
        X_image = X_image.reshape(1, -1).astype('float32') / 255.0
        embedding_image = self.forward(X_image)
        distances = np.sum(np.square(X_train - embedding_image), axis=1)
        nearest_neighbor_idx = np.argmin(distances)

        return y_train[nearest_neighbor_idx]

    def save_model_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2
            }, f)

    def load_model_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
        

input_size = 64  
hidden_size = 32  
output_size = 16 

epochs=100
model = DeepLearningModel(input_size, hidden_size, output_size)
model.train(train_triplets, epochs)

accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

filename = os.path.join("q3","model.pkl")
model.save_model_pickle(filename)