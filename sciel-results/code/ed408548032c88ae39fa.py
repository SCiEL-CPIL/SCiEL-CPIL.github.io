"""
Strategy: Kernel Ridge Regression with Composite Kernels and Hyperparameter Optimization.
This approach uses a sum of RBF, Polynomial, and Periodic kernels to capture a wide range of
functional behaviors (smooth variations, polynomial growth, and periodicity) without manual feature selection.
Hyperparameters are selected via a small grid search using 5-fold cross-validation on the training set.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 325.0  # Safety margin
    
    d = X_train.shape[1]
    n = X_train.shape[0]
    
    # Normalize inputs to [0, 1] for better kernel conditioning
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    y_var = np.var(y_train)
    
    # Kernel definitions
    def kernel_rbf(X1, X2, length_scale):
        dists = cdist(X1, X2, 'euclidean') / length_scale
        return np.exp(-0.5 * dists**2)
        
    def kernel_poly(X1, X2, degree, gamma):
        # (gamma * <x, x'> + 1)^degree
        # gamma is set to 1.0 for simplicity in normalized space
        dots = X1 @ X2.T
        return (gamma * dots + 1.0) ** degree
        
    def kernel_periodic(X1, X2, period):
        # sin^2(pi * |x - x'| / period)
        # Using Euclidean distance for simplicity in multi-dim
        dists = cdist(X1, X2, 'euclidean')
        return np.sin(np.pi * dists / period) ** 2

    def composite_kernel(X1, X2, params):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        if params.get('rbf_w', 0) > 0:
            K += params['rbf_w'] * kernel_rbf(X1, X2, params.get('rbf_ls', 1.0))
        if params.get('poly_w', 0) > 0:
            K += params['poly_w'] * kernel_poly(X1, X2, params.get('poly_deg', 2), 1.0)
        if params.get('per_w', 0) > 0:
            K += params['per_w'] * kernel_periodic(X1, X2, params.get('per_period', 1.0))
        return K

    # Hyperparameter Grid
    # We test a few combinations to balance speed and accuracy
    rbf_weights = [0.1, 1.0, 10.0]
    rbf_lengths = [0.1, 0.5, 1.0]
    poly_weights = [0.1, 1.0]
    poly_degs = [2, 3]
    per_weights = [0.1, 1.0]
    per_periods = [0.5, 1.0, 2.0]
    
    # Regularization lambda for KRR
    lambdas = [1e-6, 1e-4, 1e-2]
    
    # To keep time low, we don't test the full cartesian product of all params.
    # We test specific "archetypes" + some variations.
    candidate_params = []
    
    # 1. RBF dominant
    for lw in rbf_weights:
        for ls in rbf_lengths:
            for lam in lambdas:
                candidate_params.append({
                    'rbf_w': lw, 'rbf_ls': ls,
                    'poly_w': 0.0, 'poly_deg': 0,
                    'per_w': 0.0, 'per_period': 0.0,
                    'lam': lam
                })
                
    # 2. Poly dominant
    for pw in poly_weights:
        for pd in poly_degs:
            for lam in lambdas:
                candidate_params.append({
                    'rbf_w': 0.0, 'rbf_ls': 1.0,
                    'poly_w': pw, 'poly_deg': pd,
                    'per_w': 0.0, 'per_period': 1.0,
                    'lam': lam
                })
                
    # 3. Periodic dominant
    for pw in per_weights:
        for pp in per_periods:
            for lam in lambdas:
                candidate_params.append({
                    'rbf_w': 0.0, 'rbf_ls': 1.0,
                    'poly_w': 0.0, 'poly_deg': 0,
                    'per_w': pw, 'per_period': pp,
                    'lam': lam
                })
                
    # 4. Composite (RBF + Poly)
    for lw in [1.0]:
        for ls in [0.5, 1.0]:
            for pw in [1.0]:
                for pd in [2]:
                    for lam in [1e-4, 1e-2]:
                        candidate_params.append({
                            'rbf_w': lw, 'rbf_ls': ls,
                            'poly_w': pw, 'poly_deg': pd,
                            'per_w': 0.0, 'per_period': 1.0,
                            'lam': lam
                        })
                        
    # 5. Composite (RBF + Periodic)
    for lw in [1.0]:
        for ls in [0.5, 1.0]:
            for pw in [1.0]:
                for pp in [1.0, 2.0]:
                    for lam in [1e-4, 1e-2]:
                        candidate_params.append({
                            'rbf_w': lw, 'rbf_ls': ls,
                            'poly_w': 0.0, 'poly_deg': 0,
                            'per_w': pw, 'per_period': pp,
                            'lam': lam
                        })

    # 5-Fold Cross Validation
    # Create folds
    indices = np.arange(n)
    # Shuffle indices for CV
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    best_score = np.inf
    best_params = candidate_params[0]
    
    # Pre-compute kernel matrices for training set for each candidate? 
    # No, that's too memory heavy if we have many candidates. 
    # We compute K_train once for the "structure" if weights were fixed, but weights change.
    # Actually, K depends on params. 
    # To speed up, we can pre-compute base kernels: K_rbf_ls, K_poly_deg, K_per_per
    # But params vary in length_scale/degree/period.
    # Given n=200, computing 200x200 kernel is fast (~1ms).
    # We have ~50-100 candidates. 5 folds. 500 kernel computations. 500ms total. Acceptable.
    
    for params in candidate_params:
        if time.time() > deadline - 10:
            break
            
        cv_scores = []
        lam = params['lam']
        
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            X_tr = X_norm[train_idx]
            y_tr = y_centered[train_idx]
            X_val = X_norm[val_idx]
            y_val = y_centered[val_idx]
            
            K_train = composite_kernel(X_tr, X_tr, params)
            K_train += lam * np.eye(len(train_idx))
            
            try:
                # Solve (K + lam*I) alpha = y
                # Use Cholesky
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            except np.linalg.LinAlgError:
                try:
                    alpha, *_ = np.linalg.lstsq(K_train, y_tr, rcond=None)
                except:
                    cv_scores.append(np.inf)
                    continue
            
            K_val_train = composite_kernel(X_val, X_tr, params)
            y_pred = K_val_train @ alpha
            
            mse = np.mean((y_pred - y_val)**2)
            cv_scores.append(mse)
            
        avg_cv_mse = np.mean(cv_scores)
        
        if avg_cv_mse < best_score:
            best_score = avg_cv_mse
            best_params = params

    # Final fit with best params
    lam = best_params['lam']
    K_train = composite_kernel(X_norm, X_norm, best_params)
    K_train += lam * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
    except np.linalg.LinAlgError:
        alpha, *_ = np.linalg.lstsq(K_train, y_centered, rcond=None)
        
    def predict(X):
        X_new_norm = (X - min_vals) / ranges
        K_s = composite_kernel(X_new_norm, X_norm, best_params)
        y_pred = y_mean + K_s @ alpha
        return y_pred

    # Verify
    pred_train = predict(X_train)
    if not np.all(np.isfinite(pred_train)):
        # Fallback to linear
        X_with_ones = np.column_stack([np.ones(n), X_train])
        w, *_ = np.linalg.lstsq(X_with_ones, y_train, rcond=None)
        def predict_fallback(X):
            X_new_with_ones = np.column_stack([np.ones(X.shape[0]), X])
            return X_new_with_ones @ w
        return predict_fallback
        
    return predict

# EVOLVE-BLOCK-END
