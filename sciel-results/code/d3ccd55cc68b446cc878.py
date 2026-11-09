"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
import scipy.sparse as sp


def _build_features(X, d, max_poly_deg=3, include_trig=True, include_exp=True, include_log=True):
    """Build a rich feature matrix with polynomial, trig, exp, log, and interaction terms."""
    n = X.shape[0]
    features = [np.ones(n)]
    
    # Polynomial terms up to degree max_poly_deg
    for deg in range(1, max_poly_deg + 1):
        for i in range(d):
            for j in range(d):
                if deg == 1:
                    if i == j:
                        features.append(X[:, i] ** deg)
                elif deg == 2:
                    if i <= j:
                        features.append(X[:, i] ** i * X[:, j] ** (deg - i) if i < d else X[:, i] ** deg)
                        if i == j:
                            features.append(X[:, i] ** deg)
                        else:
                            features.append(X[:, i] * X[:, j])
                else:
                    # For higher degrees, only include pure powers and some interactions
                    if i == j:
                        features.append(X[:, i] ** deg)
    
    # Add trigonometric features
    if include_trig:
        for i in range(d):
            features.append(np.sin(X[:, i]))
            features.append(np.cos(X[:, i]))
            features.append(np.sin(2 * X[:, i]))
            features.append(np.cos(2 * X[:, i]))
    
    # Add exponential features (careful with overflow)
    if include_exp:
        for i in range(d):
            # Clip to avoid overflow
            x_clip = np.clip(X[:, i], -50, 50)
            features.append(np.exp(x_clip))
            features.append(np.exp(-x_clip))
    
    # Add logarithmic features (only where positive)
    if include_log:
        for i in range(d):
            x_pos = np.where(X[:, i] > 0, X[:, i], 1e-10)
            features.append(np.log(x_pos))
    
    # Add interaction terms (degree 2 cross terms)
    for i in range(d):
        for j in range(i + 1, d):
            features.append(X[:, i] * X[:, j])
    
    # Add squared terms
    for i in range(d):
        features.append(X[:, i] ** 2)
    
    # Add cube terms
    for i in range(d):
        features.append(X[:, i] ** 3)
    
    feat_matrix = np.column_stack(features)
    
    # Replace any NaN/inf with 0
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feat_matrix


def fit(X_train, y_train):
    """
    Fit a sparse linear model on a rich set of basis functions.
    Uses Lasso-like regularization via iterative feature selection.
    """
    d = X_train.shape[1]
    n = X_train.shape[0]
    
    # Build rich feature set
    F = _build_features(X_train, d, max_poly_deg=3, include_trig=True, include_exp=True, include_log=True)
    
    # Normalize features for better conditioning
    col_means = F.mean(axis=0)
    col_stds = F.std(axis=0)
    col_stds[col_stds < 1e-10] = 1.0
    F_norm = (F - col_means) / col_stds
    
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    # Use ridge regression with multiple alpha values and pick best via cross-validation-like approach
    # For speed, use a simple approach: try several alphas and pick the one with best training MSE
    best_w = None
    best_mse = np.inf
    
    alphas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    for alpha in alphas:
        # Ridge regression: (F^T F + alpha*I)^-1 F^T y
        A = F_norm.T @ F_norm + alpha * np.eye(F_norm.shape[1])
        b = F_norm.T @ y_norm
        try:
            w = np.linalg.solve(A, b)
            pred_norm = F_norm @ w
            mse = np.mean((pred_norm - y_norm) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_w = w
        except np.linalg.LinAlgError:
            continue
    
    if best_w is None:
        # Fallback: simple OLS
        w, *_ = np.linalg.lstsq(F_norm, y_norm, rcond=None)
        best_w = w
    
    # Build predict function
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F_test = _build_features(X, d, max_poly_deg=3, include_trig=True, include_exp=True, include_log=True)
        F_test_norm = (F_test - col_means) / col_stds
        pred_norm = F_test_norm @ best_w
        pred = pred_norm * y_std + y_mean
        return pred.flatten()
    
    return predict
# EVOLVE-BLOCK-END
