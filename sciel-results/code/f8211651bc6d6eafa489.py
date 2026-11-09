"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    # Feature basis for 1D input
    def make_features(X_1d):
        x = X_1d.flatten()
        # Polynomial terms up to degree 4
        poly = [x ** k for k in range(5)]
        # Trigonometric terms
        trig = [np.sin(k * x) for k in range(1, 5)] + [np.cos(k * x) for k in range(1, 5)]
        # Exponential terms (careful with overflow)
        exp_terms = [np.exp(k * x) for k in [0.1, 0.5, 1.0, -0.1, -0.5, -1.0]]
        # Reciprocal terms (handle zero)
        x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)
        rec = [1.0 / x_safe, 1.0 / (x_safe ** 2)]
        
        return np.column_stack(poly + trig + exp_terms + rec)
    
    # Try ridge regression with different lambda values
    best_w = None
    best_score = np.inf
    
    for lam in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
        if time.time() > deadline:
            break
            
        try:
            F = make_features(X)
            # Ridge regression: solve (F^T F + lambda * I) w = F^T y
            A = F.T @ F + lam * np.eye(F.shape[1])
            b = F.T @ y
            w = np.linalg.solve(A, b)
            
            # Compute training MSE
            pred = F @ w
            mse = np.mean((pred - y) ** 2)
            var_y = np.var(y)
            if var_y > 1e-10:
                nrmse = mse / var_y
            else:
                nrmse = mse
            
            if nrmse < best_score:
                best_score = nrmse
                best_w = w
        except Exception:
            continue
    
    # If ridge regression failed, fall back to simple linear regression
    if best_w is None:
        try:
            x = X.flatten()
            A = np.column_stack([np.ones(n), x])
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            
            def predict_linear(X_pred):
                x_pred = np.asarray(X_pred, dtype=float).flatten()
                A_pred = np.column_stack([np.ones(len(x_pred)), x_pred])
                return A_pred @ w
            return predict_linear
        except Exception:
            # Last resort: return mean
            y_mean = np.mean(y)
            return lambda X: np.full(np.asarray(X).shape[0], y_mean)
    
    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float)
        F_pred = make_features(X_pred)
        return F_pred @ best_w
    
    return predict
# EVOLVE-BLOCK-END
