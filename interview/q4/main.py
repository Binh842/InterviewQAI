from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import datasets
from PIL import Image
import io

app = FastAPI()

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

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']

    def predict(self, X_image, X_train, y_train):
        X_image = X_image.reshape(1, -1).astype('float32') / 255.0
        embedding_image = self.forward(X_image)
        X_train_embeddings = self.forward(X_train)
        distances = np.sum(np.square(X_train_embeddings - embedding_image), axis=1)
        nearest_neighbor_idx = np.argmin(distances)
        return y_train[nearest_neighbor_idx]

# Khởi tạo mô hình
input_size = 64  
hidden_size = 32  
output_size = 16 

model = DeepLearningModel(input_size, hidden_size, output_size)
model.load('model.pkl')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc ảnh từ file
        image = Image.open(io.BytesIO(await file.read())).convert('L')
        image = np.array(image) / 255.0  # Tiền xử lý ảnh

        # Load dữ liệu đào tạo
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        data = data / 255

        X_train, _, y_train, _ = train_test_split(
            data, digits.target, test_size=0.2, shuffle=False
        )

        # Dự đoán
        prediction = model.predict(image, X_train, y_train)
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return {"status": "Model is loaded and ready to use."}
