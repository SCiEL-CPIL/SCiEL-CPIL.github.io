"""
Strategy: Hybrid Feature Expansion with Ridge Regression.

This approach constructs a rich set of basis functions for 1D inputs:
1. Polynomials up to degree 6.
2. Trigonometric functions (sin, cos) for low frequencies.
3. Exponential and Logarithmic functions.
4. Radial Basis Functions (Gaussians) centered at training data quantiles.

We use Ridge Regression to fit the coefficients, regularizing to prevent
overfitting and handle potential collinearity. A small grid search over
the regularization strength (alpha) and RBF width (sigma) is performed
using 5-fold cross-validation to select the best model for each equation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qnorm

def fit(X_train, y_train):
    start_time = time.time()
    time_limit = 280.0  # Leave 20s buffer for safety
    
    # Ensure X is 1D array of shape (N, 1) or (N,)
    X = np.asarray(X_train)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y_train).ravel()
    
    x = X[:, 0]
    
    # Generate candidate feature sets based on different RBF widths and polynomial degrees
    # We will evaluate a few combinations of (poly_degree, rbf_width, alpha)
    
    # Precompute quantiles for RBF centers
    # Using 10 RBF centers spaced by quantiles ensures coverage
    rbf_centers = np.quantile(x, np.linspace(0.1, 0.9, 10))
    
    def make_features(x_val, poly_deg, rbf_width):
        """
        Construct the feature matrix for a given input x_val.
        """
        # 1. Polynomials
        # x^0, x^1, ..., x^poly_deg
        poly_cols = [np.ones_like(x_val)]
        for k in range(1, poly_deg + 1):
            poly_cols.append(x_val ** k)
        
        # 2. Trigonometric
        # sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x)
        trig_cols = []
        for k in range(1, 4):
            trig_cols.append(np.sin(k * x_val))
            trig_cols.append(np.cos(k * x_val))
        
        # 3. Exponential / Log
        # exp(x), exp(-x), log(|x| + 1e-6)
        exp_cols = [
            np.exp(np.clip(x_val, -10, 10)),  # Clip to avoid overflow
            np.exp(-np.clip(x_val, -10, 10)),
            np.log(np.abs(x_val) + 1e-6)
        ]
        
        # 4. RBFs
        # exp(-0.5 * ((x - c) / width)^2)
        rbf_cols = []
        if rbf_width > 0:
            for c in rbf_centers:
                rbf_cols.append(np.exp(-0.5 * ((x_val - c) / rbf_width) ** 2))
        
        cols = poly_cols + trig_cols + exp_cols + rbf_cols
        return np.column_stack(cols)
    
    # Hyperparameters to search
    # alpha: regularization strength
    # rbf_width: width of the Gaussian RBFs
    # poly_deg: max degree of polynomial
    candidates = [
        (1e-8, 0.5, 4),
        (1e-8, 1.0, 4),
        (1e-8, 2.0, 4),
        (1e-6, 0.5, 4),
        (1e-6, 1.0, 4),
        (1e-6, 2.0, 4),
        (1e-4, 0.5, 4),
        (1e-4, 1.0, 4),
        (1e-4, 2.0, 4),
        (1e-2, 0.5, 4),
        (1e-2, 1.0, 4),
        (1e-2, 2.0, 4),
        (1e-8, 0.5, 6),
        (1e-8, 1.0, 6),
        (1e-6, 0.5, 6),
        (1e-6, 1.0, 6),
    ]
    
    # 5-Fold Cross Validation
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    best_score = np.inf
    best_params = candidates[0]
    
    for alpha, width, deg in candidates:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Precompute features for training set for this config
        # Note: We recompute features inside the loop for simplicity, 
        # but since it's 200 points, it's very fast.
        X_feat_train = make_features(x, deg, width)
        
        cv_errors = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            X_tr = X_feat_train[train_idx]
            y_tr = y[train_idx]
            X_val = X_feat_train[val_idx]
            y_val = y[val_idx]
            
            # Ridge Regression
            # min ||Xw - y||^2 + alpha * ||w||^2
            # (X^T X + alpha * I) w = X^T y
            try:
                A = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
                b = X_tr.T @ y_tr
                w = np.linalg.solve(A, b)
                pred_val = X_val @ w
                mse = np.mean((pred_val - y_val) ** 2)
                cv_errors.append(mse)
            except np.linalg.LinAlgError:
                cv_errors.append(np.inf)
        
        mean_cv_error = np.mean(cv_errors)
        if mean_cv_error < best_score:
            best_score = mean_cv_error
            best_params = (alpha, width, deg)
    
    # Fit final model with best parameters
    alpha, width, deg = best_params
    X_feat_final = make_features(x, deg, width)
    
    # Use a slightly smaller alpha for final fit if CV error was very low? 
    # Or just stick with the selected alpha. Sticking is safer.
    try:
        A = X_feat_final.T @ X_feat_final + alpha * np.eye(X_feat_final.shape[1])
        b = X_feat_final.T @ y
        w_final = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular (should not happen with ridge)
        w_final, *_ = np.linalg.lstsq(X_feat_final, y, rcond=None)

    def predict(X_test):
        X_t = np.asarray(X_test)
        if X_t.ndim == 1:
            X_t = X_t.reshape(-1, 1)
        x_t = X_t[:, 0]
        X_feat_test = make_features(x_t, deg, width)
        preds = X_feat_test @ w_final
        # Ensure finite outputs
        preds = np.nan_to_num(preds, nan=0.0, posinf=1e10, neginf=-1e10)
        return preds

    return predict
# EVOLVE-BLOCK-END
