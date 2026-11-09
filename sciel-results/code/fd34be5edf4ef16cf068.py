"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def fit(X_train, y_train):
    """
    Fit a regression model using a structured library of candidate functions
    with least-squares selection via cross-validation.
    """
    n, d = X_train.shape
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 300s limit
    
    # Extract the single input column (all equations have 1 input)
    x = X_train[:, 0]
    y = y_train.copy()
    
    # Check for constant or near-constant y
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    # Build a library of candidate basis functions
    def build_library(x_vals):
        """Return a matrix of candidate basis functions evaluated at x_vals."""
        x_abs = np.abs(x_vals)
        x_safe = np.where(np.abs(x_vals) < 1e-10, 1e-10, x_vals)
        
        candidates = [
            np.ones_like(x_vals),                          # 1
            x_vals,                                         # x
            x_vals ** 2,                                    # x^2
            x_vals ** 3,                                    # x^3
            x_vals ** 4,                                    # x^4
            x_vals ** 5,                                    # x^5
            x_vals ** 6,                                    # x^6
            x_vals ** 7,                                    # x^7
            x_vals ** 8,                                    # x^8
            x_vals ** 10,                                   # x^10
            np.sin(x_vals),                                 # sin(x)
            np.cos(x_vals),                                 # cos(x)
            np.sin(2 * x_vals),                             # sin(2x)
            np.cos(2 * x_vals),                             # cos(2x)
            np.sin(3 * x_vals),                             # sin(3x)
            np.cos(3 * x_vals),                             # cos(3x)
            np.sin(4 * x_vals),                             # sin(4x)
            np.cos(4 * x_vals),                             # cos(4x)
            np.sin(5 * x_vals),                             # sin(5x)
            np.cos(5 * x_vals),                             # cos(5x)
            np.sin(6 * x_vals),                             # sin(6x)
            np.cos(6 * x_vals),                             # cos(6x)
            np.sin(8 * x_vals),                             # sin(8x)
            np.cos(8 * x_vals),                             # cos(8x)
            np.sin(10 * x_vals),                            # sin(10x)
            np.cos(10 * x_vals),                            # cos(10x)
            np.log1p(x_abs),                                # log(1+|x|)
            np.log1p(x_vals + x_vals.max() + 1),            # log(x + offset)
            1.0 / (1.0 + x_vals ** 2),                      # 1/(1+x^2)
            1.0 / (1.0 + 2.0 * x_vals ** 2),                # 1/(1+2x^2)
            1.0 / (1.0 + 5.0 * x_vals ** 2),                # 1/(1+5x^2)
            1.0 / x_safe,                                   # 1/x
            1.0 / (x_safe ** 2),                            # 1/x^2
            x_vals / (1.0 + x_vals ** 2),                   # x/(1+x^2)
            x_vals / (1.0 + 2.0 * x_vals ** 2),             # x/(1+2x^2)
            x_vals ** 2 / (1.0 + x_vals ** 2),              # x^2/(1+x^2)
            np.tanh(x_vals),                                # tanh(x)
            np.tanh(2 * x_vals),                            # tanh(2x)
            np.tanh(0.5 * x_vals),                          # tanh(0.5x)
            np.exp(-x_vals ** 2 / 2.0),                     # exp(-x^2/2)
            np.exp(-x_vals ** 2),                           # exp(-x^2)
            np.exp(-x_vals ** 2 / 4.0),                     # exp(-x^2/4)
            np.exp(-np.abs(x_vals)),                        # exp(-|x|)
            np.exp(-0.5 * np.abs(x_vals)),                  # exp(-0.5|x|)
            np.exp(-2.0 * np.abs(x_vals)),                  # exp(-2|x|)
            np.sqrt(x_abs),                                 # sqrt(|x|)
            np.sqrt(x_vals + x_vals.max() + 1),             # sqrt(x + offset)
            x_vals * np.sin(x_vals),                        # x*sin(x)
            x_vals * np.cos(x_vals),                        # x*cos(x)
            x_vals ** 2 * np.sin(x_vals),                   # x^2*sin(x)
            x_vals ** 2 * np.cos(x_vals),                   # x^2*cos(x)
            np.sin(x_vals) / x_safe,                        # sin(x)/x
            np.cos(x_vals) - 1.0,                           # cos(x)-1
            x_vals ** 3 / (1.0 + x_vals ** 2),              # x^3/(1+x^2)
            np.log(x_abs + 1e-10),                          # log(|x|)
            x_vals * np.log1p(x_abs),                       # x*log(1+|x|)
        ]
        
        # Stack into matrix
        lib = np.column_stack(candidates)
        # Replace any non-finite values
        lib = np.where(np.isfinite(lib), lib, 0.0)
        return lib
    
    # Precompute library for training data
    lib_train = build_library(x)
    
    # Cross-validation to select best subset of basis functions
    def cv_select(k, n_folds=5):
        """Select best k basis functions via CV."""
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        fold_size = n // n_folds
        scores = []
        
        for i in range(n_folds):
            val_idx = idx[i * fold_size:(i + 1) * fold_size]
            train_idx = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
            
            X_tr = lib_train[train_idx]
            y_tr = y[train_idx]
            X_val = lib_train[val_idx]
            y_val = y[val_idx]
            
            # Use ridge regression for stability
            alpha = 1e-6
            try:
                A = X.T @ X + alpha * np.eye(X.shape[1])
                b = X.T @ y_tr
                w = np.linalg.solve(A, b)
                pred = X_val @ w
                if not np.all(np.isfinite(pred)):
                    return np.inf
                mse = np.mean((pred - y_val) ** 2)
                scores.append(mse)
            except Exception:
                return np.inf
        
        return np.mean(scores)
    
    # Try different numbers of basis functions
    best_k = 1
    best_cv_score = np.inf
    
    n_features = lib_train.shape[1]
    # Try k = 1 to min(20, n_features)
    for k in range(1, min(21, n_features + 1)):
        if time.time() > deadline - 60.0:
            break
        try:
            score = cv_select(k)
            if score < best_cv_score:
                best_cv_score = score
                best_k = k
        except Exception:
            continue
    
    # Now select which specific basis functions to use
    # Use a greedy forward selection
    def greedy_select(k_max, n_folds=5):
        """Greedy forward selection of basis functions."""
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        fold_size = n // n_folds
        
        selected = [0]  # Always include intercept
        remaining = list(range(1, n_features))
        
        for step in range(k_max - 1):
            if time.time() > deadline - 30.0:
                break
            best_feat = None
            best_score = np.inf
            
            for feat in remaining:
                candidate = selected + [feat]
                # Evaluate CV score
                scores = []
                for i in range(n_folds):
                    val_idx = idx[i * fold_size:(i + 1) * fold_size]
                    train_idx = np.concatenate([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
                    
                    X_tr = lib_train[train_idx, candidate]
                    y_tr = y[train_idx]
                    X_val = lib_train[val_idx, candidate]
                    y_val = y[val_idx]
                    
                    alpha = 1e-6
                    try:
                        A = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
                        b = X_tr.T @ y_tr
                        w = np.linalg.solve(A, b)
                        pred = X_val @ w
                        if not np.all(np.isfinite(pred)):
                            scores = [np.inf]
                            break
                        mse = np.mean((pred - y_val) ** 2)
                        scores.append(mse)
                    except Exception:
                        scores = [np.inf]
                        break
                
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_feat = feat
            
            if best_feat is None:
                break
            selected.append(best_feat)
            remaining.remove(best_feat)
        
        return selected
    
    # Perform greedy selection
    selected_features = greedy_select(best_k)
    
    # Fit final model on all data
    X_final = lib_train[:, selected_features]
    alpha = 1e-6
    try:
        A = X_final.T @ X_final + alpha * np.eye(X_final.shape[1])
        b = X_final.T @ y
        w = np.linalg.solve(A, b)
    except Exception:
        # Fallback to simple linear
        w = np.linalg.lstsq(X_final, y, rcond=None)[0]
    
    def predict(X):
        x_pred = X[:, 0]
        lib_pred = build_library(x_pred)
        return lib_pred[:, selected_features] @ w
    
    # Verify finite outputs
    test_pred = predict(X_train[:10])
    if not np.all(np.isfinite(test_pred)):
        # Fallback: simple polynomial
        def predict_fallback(X):
            x_pred = X[:, 0]
            return y_mean + np.zeros(len(x_pred))
        return predict_fallback
    
    return predict
# EVOLVE-BLOCK-END
