import numpy as np
import time

def fit(X_train, y_train):
    """
    Fits a small neural network to the training data.
    Returns a predict callable.
    """
    X = X_train
    y = y_train
    
    # Ensure 2D input for consistency, though task says 1D inputs
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Standardize inputs and outputs for better NN training
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    # Avoid division by zero
    x_std[x_std == 0] = 1.0
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
        
    X_norm = (X - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    
    # Define a simple Neural Network using NumPy
    # Architecture: 1 -> 32 -> 32 -> 1
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def relu(z):
        return np.maximum(0, z)
    
    def relu_grad(z):
        return (z > 0).astype(float)
    
    # Initialize weights with He initialization
    np.random.seed(42)
    W1 = np.random.randn(n_features, 32) * np.sqrt(2.0 / n_features)
    b1 = np.zeros((1, 32))
    W2 = np.random.randn(32, 32) * np.sqrt(2.0 / 32)
    b2 = np.zeros((1, 32))
    W3 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
    b3 = np.zeros((1, 1))
    
    # Hyperparameters
    learning_rate = 0.005
    epochs = 1000
    batch_size = 32 # Mini-batch or full batch? 200 samples is small, full batch is stable
    # Let's use full batch for simplicity and stability on small data
    
    start_time = time.time()
    deadline = start_time + 250 # Safety margin for 300s limit
    
    # Training loop
    for epoch in range(epochs):
        if time.time() > deadline:
            break
            
        # Forward pass
        z1 = X_norm @ W1 + b1
        a1 = relu(z1)
        
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        
        z3 = a2 @ W3 + b3
        a3 = z3 # Linear output
        
        # Loss: MSE
        error = a3 - y_norm.reshape(-1, 1)
        # MSE = 0.5 * mean(error^2)
        
        # Backward pass
        m = n_samples
        dz3 = error / m
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        da2 = dz3 @ W3.T
        dz2 = da2 * relu_grad(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = X_norm.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights with Adam-like simple momentum or just SGD
        # Simple SGD with small LR is often enough for this small problem
        # Let's use a simple momentum update for better convergence
        # To keep code short, we'll stick to vanilla SGD with a slightly higher LR if needed, 
        # but 0.005 is usually safe. Let's try 0.01.
        
        # Actually, let's implement a simple momentum for robustness
        # But for brevity and speed, let's just do SGD.
        # If it doesn't converge, we can increase epochs.
        
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        
    def predict(X_pred):
        X_pred = np.asarray(X_pred)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(-1, 1)
        
        X_norm_pred = (X_pred - x_mean) / x_std
        
        # Forward pass
        z1 = X_norm_pred @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = a2 @ W3 + b3
        a3 = z3
        
        # Denormalize
        y_pred = a3.flatten() * y_std + y_mean
        
        # Ensure finite
        y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)
        
        return y_pred

    return predict
