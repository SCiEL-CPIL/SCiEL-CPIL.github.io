import numpy as np
import time
from scipy.linalg import solve

def _create_features(x):
    """
    Create a rich set of features for 1D input x (normalized to [-1, 1]).
    """
    features = []
    
    # Polynomials up to degree 10
    for i in range(11):
        features.append(x**i)
        
    # Trigonometric functions
    # Frequencies scaled to the normalized range
    for k in range(1, 6):
        features.append(np.sin(k * np.pi * x))
        features.append(np.cos(k * np.pi * x))
        
    # Exponential functions
    features.append(np.exp(-x**2))
    features.append(np.exp(x))
    features.append(np.exp(-x))
    features.append(np.exp(-x**2 / 2))
    features.append(np.exp(-x**2 / 0.5))
    
    # Rational functions
    features.append(1.0 / (1.0 + x**2))
    features.append(1.0 / (1.0 + np.abs(x)))
    
    # Absolute value and related
    features.append(np.abs(x))
    features.append(x * np.abs(x))
    
    # Square root (safe)
    # Since x is in [-1, 1], sqrt(x) is only real for x>=0. 
    # We can use sqrt(|x|) or handle carefully. 
    # Let's add sqrt(|x|) as a feature, it's continuous.
    features.append(np.sqrt(np.abs(x)))
    
    return np.column_stack(features)

def _fit_ridge_cv(X, y, lambdas, n_folds=5):
    """
    Fit Ridge Regression with Cross-Validation to select best lambda.
    X: (n_samples, n_features)
    y: (n_samples,)
    lambdas: list of lambda values
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split into folds
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])
    
    best_lambda = lambdas[0]
    best_cv_error = np.inf
    best_coeffs = None
    
    for lam in lambdas:
        cv_errors = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Solve Ridge: (X^T X + lam * I) w = X^T y
            # Add small ridge to identity for stability even if lam is small
            A = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
            b = X_train.T @ y_train
            
            try:
                w = solve(A, b)
            except Exception:
                w = np.linalg.lstsq(A, b, rcond=None)[0]
                
            y_pred_val = X_val @ w
            mse = np.mean((y_pred_val - y_val)**2)
            cv_errors.append(mse)
            
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_cv_error:
            best_cv_error = mean_cv_error
            best_lambda = lam
            # Store coeffs for refitting later? Or just refit at the end.
            
    # Refit on all data with best_lambda
    A = X.T @ X + best_lambda * np.eye(X.shape[1])
    b = X.T @ y
    try:
        w = solve(A, b)
    except Exception:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        
    return w, best_lambda, best_cv_error

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    X = np.asarray(X_train, dtype=float).flatten()
    y = np.asarray(y_train, dtype=float).flatten()
    
    # Normalize X to [-1, 1]
    x_min = np.min(X)
    x_max = np.max(X)
    x_range = x_max - x_min
    if x_range < 1e-8:
        x_norm = np.zeros_like(X)
    else:
        x_norm = 2.0 * (X - x_min) / x_range - 1.0
    
    # Create features
    try:
        F = _create_features(x_norm)
    except Exception:
        # Fallback to simple polynomial if feature creation fails
        F = np.column_stack([np.ones_like(x_norm), x_norm, x_norm**2])
        
    # Standardize features to have zero mean and unit variance for better conditioning
    F_mean = np.mean(F, axis=0)
    F_std = np.std(F, axis=0)
    F_std[F_std < 1e-8] = 1.0
    F_norm = (F - F_mean) / F_std
    
    # Standardize y for better conditioning of the regression problem
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Define lambda range for Ridge Regression
    # Log-scale from 1e-5 to 1e5
    lambdas = np.logspace(-5, 5, 21)
    
    best_w = None
    best_score = np.inf
    best_model_type = "mean"
    
    # Fit Ridge with CV
    try:
        w, best_lambda, cv_mse = _fit_ridge_cv(F_norm, y_norm, lambdas)
        
        # Calculate training score for the selected model
        y_pred_norm = F_norm @ w
        y_pred = y_pred_norm * y_std + y_mean
        mse = np.mean((y_pred - y)**2)
        var_y = np.var(y)
        if var_y < 1e-10:
            var_y = 1.0
        train_score = mse / var_y
        
        # Use CV error as a proxy for generalization, but also check train score
        # If train score is very high, something might be wrong
        if train_score < 10.0:  # Sanity check
            best_w = w
            best_score = cv_mse / (y_std**2)  # Scale back to original units
            best_model_type = "ridge"
    except Exception:
        pass
    
    # Fallback 1: Simple Linear Regression (Polynomial degree 2)
    if best_model_type != "ridge":
        try:
            # Use just first few features (polynomials)
            F_poly = F_norm[:, :3] # 1, x, x^2
            # Simple Ridge with small lambda
            lam = 1e-3
            A = F_poly.T @ F_poly + lam * np.eye(F_poly.shape[1])
            b = F_poly.T @ y_norm
            w_poly = solve(A, b)
            
            y_pred_norm = F_poly @ w_poly
            y_pred = y_pred_norm * y_std + y_mean
            mse = np.mean((y_pred - y)**2)
            var_y = np.var(y)
            if var_y < 1e-10:
                var_y = 1.0
            score = mse / var_y
            
            if score < best_score:
                best_w = w_poly
                best_score = score
                best_model_type = "poly2"
        except Exception:
            pass
            
    # Fallback 2: Mean
    if best_model_type not in ["ridge", "poly2"]:
        best_model_type = "mean"
        best_score = 1.0 # Normalized MSE of mean predictor is 1.0

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            x = X_test
        else:
            x = X_test[:, 0]
            
        # Normalize
        if x_range < 1e-8:
            x_norm = np.zeros_like(x)
        else:
            x_norm = 2.0 * (x - x_min) / x_range - 1.0
            
        if best_model_type == "mean":
            return np.full(x.shape, y_mean)
            
        try:
            F_test = _create_features(x_norm)
            F_test_norm = (F_test - F_mean) / F_std
            
            if best_model_type == "ridge":
                y_pred_norm = F_test_norm @ best_w
            else: # poly2
                F_test_norm = F_test_norm[:, :3]
                y_pred_norm = F_test_norm @ best_w
                
            y_pred = y_pred_norm * y_std + y_mean
            return y_pred
        except Exception:
            # If prediction fails, return mean
            return np.full(x.shape, y_mean)

    return predict
