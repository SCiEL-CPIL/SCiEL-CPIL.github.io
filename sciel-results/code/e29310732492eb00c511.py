"""Symbolic regression via adaptive-bandwidth kernel ridge regression with RBF + polynomial + Fourier features."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_fourier_features(X, n_freqs=10, max_dim=200):
    """Build Fourier basis features: sin/cos of k * x_i for k=1..n_freqs per dimension.
    Capped at max_dim total features to avoid explosion."""
    n, d = X.shape
    parts = [np.ones((n, 1))]
    for i in range(d):
        xi = X[:, i]
        for k in range(1, n_freqs + 1):
            parts.append(np.sin(k * np.pi * xi).reshape(-1, 1))
            parts.append(np.cos(k * np.pi * xi).reshape(-1, 1))
    F = np.hstack(parts)
    if F.shape[1] > max_dim:
        F = F[:, :max_dim]
    return F

def _build_poly_features(X, max_deg=3):
    """Build polynomial features up to max_deg (constant + linear + ... + max_deg)."""
    n, d = X.shape
    parts = [np.ones((n, 1)), X.copy()]
    if max_deg >= 2:
        quad = np.zeros((n, d * (d + 1) // 2))
        idx = 0
        for i in range(d):
            for j in range(i, d):
                quad[:, idx] = X[:, i] * X[:, j]
                idx += 1
        parts.append(quad)
    if max_deg >= 3:
        cub = np.zeros((n, d * (d + 1) * (d + 2) // 6))
        idx = 0
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cub[:, idx] = X[:, i] * X[:, j] * X[:, k]
                    idx += 1
        parts.append(cub)
    return np.hstack(parts)

def fit(X_train, y_train):
    """Fit a kernel ridge regression with RBF kernel."""
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X - X_min) / X_range

    # Precompute pairwise squared distances
    diff = X_norm[:, None, :] - X_norm[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)

    # Candidate bandwidths (l2 = 1/(2*sigma^2))
    l2_candidates = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    alpha_candidates = [1e-4, 1e-3, 1e-2, 1e-1]

    # Validation split
    rng = np.random.RandomState(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    half = n // 2
    val_idx = idx[:half]
    train_idx = idx[half:]

    best_score = np.inf
    best_l2 = 1.0
    best_alpha = 1e-3

    for l2 in l2_candidates:
        K = np.exp(-l2 * sq_dist)
        for alpha in alpha_candidates:
            # Solve kernel ridge on train split
            K_train = K[np.ix_(train_idx, train_idx)]
            y_train_split = y[train_idx]
            
            # (K + alpha I) w = y
            A = K_train + alpha * np.eye(len(train_idx))
            try:
                w = np.linalg.solve(A, y_train_split)
            except np.linalg.LinAlgError:
                continue
            
            # Predict on val
            K_val = K[np.ix_(val_idx, train_idx)]
            pred_val = K_val @ w
            mse_val = np.mean((pred_val - y[val_idx]) ** 2)
            
            if mse_val < best_score:
                best_score = mse_val
                best_l2 = l2
                best_alpha = alpha

    # Final fit on all data
    K_final = np.exp(-best_l2 * sq_dist)
    A_final = K_final + best_alpha * np.eye(n)
    w_final = np.linalg.solve(A_final, y)

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_norm = (X_new - X_min) / X_range
        
        # RBF kernel between new and train
        diff_new = X_new_norm[:, None, :] - X_norm[None, :, :]
        sq_dist_new = np.sum(diff_new ** 2, axis=2)
        K_new = np.exp(-best_l2 * sq_dist_new)
        
        return K_new @ w_final

    return predict
# EVOLVE-BLOCK-END
