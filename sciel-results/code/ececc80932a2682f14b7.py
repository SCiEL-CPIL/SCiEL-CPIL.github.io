"""Rewritten program using a small neural network trained with Adam."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def fit(X_train, y_train):
    """
    Fit a small neural network (2 hidden layers, 32 units) to the data.
    Returns a predict callable.
    """
    n_samples, n_features = X_train.shape
    
    # Normalize inputs and outputs for better training
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    # Initialize weights and biases
    np.random.seed(42)
    
    # Layer 1: n_features -> 32
    W1 = np.random.randn(n_features, 32) * np.sqrt(2.0 / n_features)
    b1 = np.zeros(32)
    
    # Layer 2: 32 -> 32
    W2 = np.random.randn(32, 32) * np.sqrt(2.0 / 32)
    b2 = np.zeros(32)
    
    # Output layer: 32 -> 1
    W3 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
    b3 = np.zeros(1)
    
    # Combine parameters into a single vector for optimization
    params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2, W3.flatten(), b3])
    
    def forward(params, X):
        W1 = params[:n_features * 32].reshape(n_features, 32)
        b1 = params[n_features * 32:n_features * 32 + 32]
        W2 = params[n_features * 32 + 32:n_features * 32 + 32 + 32 * 32].reshape(32, 32)
        b2 = params[n_features * 32 + 32 + 32 * 32:n_features * 32 + 32 + 32 * 32 + 32]
        W3 = params[n_features * 32 + 32 + 32 * 32 + 32:n_features * 32 + 32 + 32 * 32 + 32 + 32].reshape(32, 1)
        b3 = params[n_features * 32 + 32 + 32 * 32 + 32 + 32:]
        
        h1 = np.tanh(X @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        out = h2 @ W3 + b3
        return out, (h1, h2)
    
    def loss_and_grad(params, X, y):
        out, (h1, h2) = forward(params, X)
        out = out.flatten()
        # MSE loss
        diff = out - y
        loss = np.mean(diff ** 2)
        
        # Backpropagation
        d_out = 2 * diff / n_samples
        
        W1 = params[:n_features * 32].reshape(n_features, 32)
        b1 = params[n_features * 32:n_features * 32 + 32]
        W2 = params[n_features * 32 + 32:n_features * 32 + 32 + 32 * 32].reshape(32, 32)
        b2 = params[n_features * 32 + 32 + 32 * 32:n_features * 32 + 32 + 32 * 32 + 32]
        W3 = params[n_features * 32 + 32 + 32 * 32 + 32:n_features * 32 + 32 + 32 * 32 + 32 + 32].reshape(32, 1)
        b3 = params[n_features * 32 + 32 + 32 * 32 + 32 + 32:]
        
        d_W3 = h2.T @ d_out
        d_b3 = np.sum(d_out, axis=0)
        
        d_h2 = d_out @ W3.T
        d_h2 *= (1 - h2 ** 2)
        
        d_W2 = h1.T @ d_h2
        d_b2 = np.sum(d_h2, axis=0)
        
        d_h1 = d_h2 @ W2.T
        d_h1 *= (1 - h1 ** 2)
        
        d_W1 = X.T @ d_h1
        d_b1 = np.sum(d_h1, axis=0)
        
        grad = np.concatenate([d_W1.flatten(), d_b1, d_W2.flatten(), d_b2, d_W3.flatten(), d_b3])
        return loss, grad
    
    # Use L-BFGS-B optimizer for better convergence
    result = minimize(
        lambda p: loss_and_grad(p, X_norm, y_norm)[0],
        params,
        jac=lambda p: loss_and_grad(p, X_norm, y_norm)[1],
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    best_params = result.x
    
    def predict(X):
        # Normalize input
        X_norm = (X - X_mean) / X_std
        out, _ = forward(best_params, X_norm)
        # Denormalize output
        return (out.flatten() * y_std + y_mean)
    
    return predict
# EVOLVE-BLOCK-END
