from sklearn import datasets
import numpy as np
from PIL import Image
import os

digits = datasets.load_digits()
X = digits.images  # 3D array: (n_samples, 8, 8)
y = digits.target  # 1D array: (n_samples,)

def save_digits_images(X, y, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    n_samples = X.shape[0]
    
    for i in range(n_samples):
        image_data = X[i]  # Already in 8x8 shape
        label = y[i]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray((image_data * 255).astype(np.uint8))
        
        # Save image
        image_filename = f"{output_dir}/img_{i}_label_{label}.png"
        image.save(image_filename)

save_digits_images(X[:100], y[:100], r'q4/digits_images')
