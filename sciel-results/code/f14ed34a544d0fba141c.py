"""Symbolic regression with adaptive feature search and time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill
    
    d = X_train.shape[1]
    n = len(X_train)
    y_var = np.var(y_train)
    y_mean = np.mean(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Normalize inputs for better numerical behavior
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-12] = 1.0
    
    def normalize(X):
        return (X - X_mean) / X_std
    
    X_norm_train = normalize(X_train)
    
    # Feature generation functions
    def poly_features(X, degree):
        """Generate polynomial features up to given degree."""
        cols = [np.ones(len(X))]
        for deg in range(1, degree + 1):
            for combo in itertools.combinations_with_replacement(range(d), deg):
                col = np.ones(len(X))
                for idx in combo:
                    col = col * X[:, idx]
                cols.append(col)
        return np.column_stack(cols)
    
    def trig_features(X):
        """Generate trigonometric features."""
        cols = [np.ones(len(X))]
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
            cols.append(np.sin(2 * X[:, j]))
            cols.append(np.cos(2 * X[:, j]))
        # Cross terms
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
                cols.append(np.cos(X[:, i]) * np.sin(X[:, j]))
        return np.column_stack(cols)
    
    def exp_features(X):
        """Generate exponential features (clipped for stability)."""
        cols = [np.ones(len(X))]
        for j in range(d):
            Xj = X[:, j]
            # Use scaled versions to avoid overflow
            cols.append(np.exp(0.5 * Xj))
            cols.append(np.exp(-0.5 * Xj))
            cols.append(np.exp(0.25 * Xj))
            cols.append(np.exp(-0.25 * Xj))
        return np.column_stack(cols)
    
    def log_features(X):
        """Generate log features (only for positive values)."""
        cols = [np.ones(len(X))]
        for j in range(d):
            Xj = X[:, j]
            # Shift to ensure positive values
            min_val = np.min(Xj)
            if min_val > 0:
                cols.append(np.log(Xj))
                cols.append(np.log(Xj ** 2))
            else:
                shift = -min_val + 1.0
                cols.append(np.log(Xj + shift))
        return np.column_stack(cols)
    
    def sqrt_features(X):
        """Generate square root features."""
        cols = [np.ones(len(X))]
        for j in range(d):
            Xj = X[:, j]
            min_val = np.min(Xj)
            if min_val >= 0:
                cols.append(np.sqrt(Xj))
                cols.append(np.sqrt(Xj ** 3))
            else:
                shift = -min_val + 1.0
                cols.append(np.sqrt(Xj + shift))
        return np.column_stack(cols)
    
    def reciprocal_features(X):
        """Generate reciprocal features."""
        cols = [np.ones(len(X))]
        for j in range(d):
            Xj = X[:, j]
            # Avoid division by zero
            safe_Xj = np.where(np.abs(Xj) < 1e-8, 1e-8, Xj)
            cols.append(1.0 / safe_Xj)
            cols.append(1.0 / (safe_Xj ** 2))
        return np.column_stack(cols)
    
    def interaction_features(X):
        """Generate pairwise interaction features."""
        cols = [np.ones(len(X))]
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.sin(X[:, i]) * np.sin(X[:, j]))
                cols.append(np.cos(X[:, i]) * np.cos(X[:, j]))
        return np.column_stack(cols)
    
    # Try different feature sets and keep the best
    feature_generators = [
        ("poly2", lambda X: poly_features(X, 2)),
        ("poly3", lambda X: poly_features(X, 3)),
        ("trig", trig_features),
        ("exp", exp_features),
        ("log", log_features),
        ("sqrt", sqrt_features),
        ("reciprocal", reciprocal_features),
        ("interaction", interaction_features),
        ("poly2+trig", lambda X: np.column_stack([poly_features(X, 2), trig_features(X)[:, 1:]])),
        ("poly2+exp", lambda X: np.column_stack([poly_features(X, 2), exp_features(X)[:, 1:]])),
        ("poly2+interaction", lambda X: np.column_stack([poly_features(X, 2), interaction_features(X)[:, 1:]])),
        ("all", lambda X: np.column_stack([
            poly_features(X, 2)[:, 1:],
            trig_features(X)[:, 1:],
            exp_features(X)[:, 1:],
            interaction_features(X)[:, 1:]
        ])),
    ]
    
    best_model = None
    best_score = np.inf
    best_predictor = None
    
    for name, gen_func in feature_generators:
        if time.time() > deadline:
            break
        
        try:
            X_feat_train = gen_func(X_norm_train)
            
            # Check for numerical issues
            if not np.all(np.isfinite(X_feat_train)):
                continue
            
            # Regularized least squares
            n_features = X_feat_train.shape[1]
            
            # Try different regularization strengths
            for alpha in [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]:
                if time.time() > deadline:
                    break
                
                # Add regularization to diagonal
                A = X_feat_train.T @ X_feat_train + alpha * np.eye(n_features)
                b = X_feat_train.T @ y_train
                
                try:
                    w = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(X_feat_train, y_train, rcond=None)[0]
                
                # Evaluate
                y_pred_train = X_feat_train @ w
                train_mse = np.mean((y_pred_train - y_train) ** 2)
                score = train_mse / y_var
                
                # Use training score as proxy (no validation set available)
                # Prefer lower score and simpler models (fewer effective features)
                effective_features = np.sum(np.abs(w) > 1e-10)
                adjusted_score = score + 0.001 * effective_features / n_features
                
                if adjusted_score < best_score:
                    best_score = adjusted_score
                    best_model = w.copy()
                    best_gen = gen_func
                    best_alpha = alpha
                    
        except Exception:
            continue
    
    # If we found a good model, use it
    if best_model is not None and best_score < 1.0:
        gen_func = best_gen
        w = best_model
        
        def predict(X):
            X_norm = normalize(X)
            X_feat = gen_func(X_norm)
            return X_feat @ w
        
        return predict
    
    # Fallback: simple linear regression on normalized features
    X_lin_train = np.column_stack([np.ones(n), X_norm_train])
    w_lin, *_ = np.linalg.lstsq(X_lin_train, y_train, rcond=None)
    
    def predict_fallback(X):
        X_norm = normalize(X)
        X_lin = np.column_stack([np.ones(len(X)), X_norm])
        return X_lin @ w_lin
    
    return predict_fallback
# EVOLVE-BLOCK-END
