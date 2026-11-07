"""Deep Feedforward Neural Network for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _relu(z):
    return np.maximum(0, z)


def _relu_grad(z):
    return (z > 0).astype(z.dtype)


def _tanh(z):
    return np.tanh(z)


def _tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


def _initialize_weights(n_in, n_hidden, n_out, seed=42):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(n_in, n_hidden) / np.sqrt(n_in)
    b1 = np.zeros(n_hidden)
    W2 = rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
    b2 = np.zeros(n_hidden)
    W3 = rng.randn(n_hidden, n_out) / np.sqrt(n_hidden)
    b3 = np.zeros(n_out)
    return W1, b1, W2, b2, W3, b3


def _forward(X, W1, b1, W2, b2, W3, b3):
    z1 = X @ W1 + b1
    a1 = _relu(z1)
    z2 = a1 @ W2 + b2
    a2 = _relu(z2)
    z3 = a2 @ W3 + b3
    return z3, z1, a1, z2, a2


def _train_nn(X_train, y_train, n_epochs=3000, lr=0.01, batch_size=32, n_hidden=64):
    n_samples, n_features = X_train.shape
    n_out = 1
    
    # Normalize inputs and outputs for better training
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    
    X_norm = (X_train - X_mean) / X_std
    y_norm = (y_train - y_mean) / y_std
    
    # Initialize weights
    W1, b1, W2, b2, W3, b3 = _initialize_weights(n_features, n_hidden, n_out, seed=42)
    
    # Adam optimizer states
    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
    mW3 = np.zeros_like(W3); vW3 = np.zeros_like(W3)
    mb3 = np.zeros_like(b3); vb3 = np.zeros_like(b3)
    
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    indices = np.arange(n_samples)
    
    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = indices[start:end]
            X_batch = X_norm[idx]
            y_batch = y_norm[idx].reshape(-1, 1)
            
            t += 1
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            # Forward pass
            z3, z1, a1, z2, a2 = _forward(X_batch, W1, b1, W2, b2, W3, b3)
            pred = z3
            
            # Loss: MSE
            error = pred - y_batch
            n_batch = len(idx)
            
            # Backward pass
            dz3 = 2 * error / n_batch
            dW3 = a2.T @ dz3
            db3 = dz3.sum(axis=0)
            da2 = dz3 @ W3.T
            dz2 = da2 * _relu_grad(z2)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ W2.T
            dz1 = da1 * _relu_grad(z1)
            dW1 = X_batch.T @ dz1
            db1 = dz1.sum(axis=0)
            
            # Adam update
            mW1 = beta1 * mW1 + (1 - beta1) * dW1; vW1 = beta2 * vW1 + (1 - beta2) * dW1**2
            mW1_hat = mW1 / (1 - beta1**t); vW1_hat = vW1 / (1 - beta2**t)
            W1 -= lr_t * mW1_hat / (np.sqrt(vW1_hat) + eps)
            
            mb1 = beta1 * mb1 + (1 - beta1) * db1; vb1 = beta2 * vb1 + (1 - beta2) * db1**2
            mb1_hat = mb1 / (1 - beta1**t); vb1_hat = vb1 / (1 - beta2**t)
            b1 -= lr_t * mb1_hat / (np.sqrt(vb1_hat) + eps)
            
            mW2 = beta1 * mW2 + (1 - beta1) * dW2; vW2 = beta2 * vW2 + (1 - beta2) * dW2**2
            mW2_hat = mW2 / (1 - beta1**t); vW2_hat = vW2 / (1 - beta2**t)
            W2 -= lr_t * mW2_hat / (np.sqrt(vW2_hat) + eps)
            
            mb2 = beta1 * mb2 + (1 - beta1) * db2; vb2 = beta2 * vb2 + (1 - beta2) * db2**2
            mb2_hat = mb2 / (1 - beta1**t); vb2_hat = vb2 / (1 - beta2**t)
            b2 -= lr_t * mb2_hat / (np.sqrt(vb2_hat) + eps)
            
            mW3 = beta1 * mW3 + (1 - beta1) * dW3; vW3 = beta2 * vW3 + (1 - beta2) * dW3**2
            mW3_hat = mW3 / (1 - beta1**t); vW3_hat = vW3 / (1 - beta2**t)
            W3 -= lr_t * mW3_hat / (np.sqrt(vW3_hat) + eps)
            
            mb3 = beta1 * mb3 + (1 - beta1) * db3; vb3 = beta2 * vb3 + (1 - beta2) * db3**2
            mb3_hat = mb3 / (1 - beta1**t); vb3_hat = vb3 / (1 - beta2**t)
            b3 -= lr_t * mb3_hat / (np.sqrt(vb3_hat) + eps)
    
    return W1, b1, W2, b2, W3, b3, X_mean, X_std, y_mean, y_std


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Try multiple architectures and pick best based on training loss
    best_loss = np.inf
    best_params = None
    
    for n_hidden in [32, 64]:
        for n_epochs in [2000]:
            try:
                params = _train_nn(X_train, y_train, n_epochs=n_epochs, lr=0.01, 
                                   batch_size=32, n_hidden=n_hidden)
                W1, b1, W2, b2, W3, b3, X_mean, X_std, y_mean, y_std = params
                
                # Compute training loss
                X_norm = (X_train - X_mean) / X_std
                z3, _, _, _, _ = _forward(X_norm, W1, b1, W2, b2, W3, b3)
                pred = (z3.flatten() * y_std + y_mean)
                loss = np.mean((pred - y_train) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = params
            except Exception:
                continue
    
    if best_params is None:
        # Fallback to linear regression
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + \
                   [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(np.asarray(X, dtype=np.float64)) @ w
    
    W1, b1, W2, b2, W3, b3, X_mean, X_std, y_mean, y_std = best_params
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - X_mean) / X_std
        z3, _, _, _, _ = _forward(X_norm, W1, b1, W2, b2, W3, b3)
        pred = z3.flatten() * y_std + y_mean
        return pred
    
    return predict
# EVOLVE-BLOCK-END
