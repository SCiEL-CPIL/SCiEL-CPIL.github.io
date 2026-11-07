# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Adaptive 1D regression using a rich feature library with greedy forward selection,
    followed by ridge regression.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    x = X_train[:, 0]
    y = y_train
    
    def generate_features(x):
        """Generate a diverse set of candidate features for 1D input x."""
        features = {}
        # Polynomials up to degree 10
        for d in range(1, 11):
            features[f'poly_{d}'] = x ** d
        # Trigonometric
        for k in [1, 2, 3, 4, 5]:
            features[f'sin_{k}'] = np.sin(k * x)
            features[f'cos_{k}'] = np.cos(k * x)
        # Exponential
        for c in [0.1, 0.5, 1.0, 2.0, -0.1, -0.5, -1.0, -2.0]:
            features[f'exp_{c}'] = np.exp(c * x)
        # Logarithmic (only for positive x)
        pos_mask = x > 0
        if np.any(pos_mask):
            log_x = np.zeros_like(x)
            log_x[pos_mask] = np.log(x[pos_mask])
            features['log'] = log_x
            features['log2'] = log_x ** 2
        # Reciprocal
        nonzero_mask = np.abs(x) > 1e-10
        if np.any(nonzero_mask):
            inv_x = np.zeros_like(x)
            inv_x[nonzero_mask] = 1.0 / x[nonzero_mask]
            features['inv'] = inv_x
            features['inv2'] = inv_x ** 2
        # Square root (for non-negative x)
        nonneg_mask = x >= 0
        if np.any(nonneg_mask):
            sqrt_x = np.zeros_like(x)
            sqrt_x[nonneg_mask] = np.sqrt(x[nonneg_mask])
            features['sqrt'] = sqrt_x
        # Tanh
        features['tanh'] = np.tanh(x)
        features['tanh2'] = np.tanh(2 * x)
        # Sigmoid-like
        features['sigmoid'] = 1.0 / (1.0 + np.exp(-x))
        # Gaussian bumps at various centers
        for center in [-3, -2, -1, 0, 1, 2, 3]:
            for width in [0.5, 1.0, 2.0]:
                features[f'gauss_{center}_{width}'] = np.exp(-((x - center) ** 2) / (2 * width ** 2))
        return features
    
    feature_dict = generate_features(x)
    feature_names = list(feature_dict.keys())
    
    # Build feature matrix with bias
    F = np.column_stack([np.ones(len(x))] + [feature_dict[name] for name in feature_names])
    
    # Greedy forward selection
    max_features = min(12, F.shape[1])
    selected = [0]  # bias always selected
    remaining = list(range(1, F.shape[1]))
    
    def compute_rss(indices):
        """Compute residual sum of squares for given feature indices."""
        F_sub = F[:, indices]
        alpha = 1e-6
        try:
            A = F_sub.T @ F_sub + alpha * np.eye(len(indices))
            b = F_sub.T @ y
            w = np.linalg.solve(A, b)
            pred = F_sub @ w
            return np.sum((y - pred) ** 2)
        except np.linalg.LinAlgError:
            return np.inf
    
    current_rss = compute_rss(selected)
    
    for step in range(max_features - 1):
        if time.time() > deadline - 30:
            break
        
        best_idx = None
        best_rss = current_rss
        
        for idx in remaining:
            trial_indices = selected + [idx]
            rss = compute_rss(trial_indices)
            if rss < best_rss:
                best_rss = rss
                best_idx = idx
        
        if best_idx is None or best_rss >= current_rss * 0.999:
            break
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_rss = best_rss
    
    # Fit final model
    F_final = F[:, selected]
    alpha = 1e-6
    try:
        A = F_final.T @ F_final + alpha * np.eye(len(selected))
        b = F_final.T @ y
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(F_final, y, rcond=None)
    
    def predict(X):
        x_test = X[:, 0]
        feature_dict_test = generate_features(x_test)
        F_test = np.column_stack([np.ones(len(x_test))] + [feature_dict_test[name] for name in feature_names])
        F_final_test = F_test[:, selected]
        return F_final_test @ w
    
    # Verify
    test_pred = predict(X_train[:5])
    if not np.all(np.isfinite(test_pred)):
        # Fallback
        def fallback_predict(X):
            x = X[:, 0]
            cols = [np.ones(len(x)), x, x**2, x**3, x**4, x**5]
            F = np.column_stack(cols)
            w_fb, *_ = np.linalg.lstsq(F, y, rcond=None)
            return F @ w_fb
        return fallback_predict
    
    return predict

# EVOLVE-BLOCK-END
