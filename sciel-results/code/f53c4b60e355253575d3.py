"""
Symbolic regression using a rich basis function expansion (polynomial, trigonometric,
exponential, logarithmic, tanh, abs) with ridge regression and adaptive regularization.
This approach captures a wide variety of functional forms without requiring local
neighborhood selection, and ridge regression provides stable coefficient estimation.
"""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _t_start)


def _rich_basis(X):
    """
    Generate a rich set of basis functions from input X (1D array).
    Includes:
    - Polynomial terms up to degree 8
    - Trigonometric functions: sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x)
    - Exponential functions: exp(x), exp(-x), exp(x/2), exp(-x/2)
    - Logarithmic: log(1+|x|), log(1+x) for x>-1
    - Tanh: tanh(x), tanh(2x)
    - Absolute value: |x|, |x|^2, |x|^3
    - Sqrt: sqrt(|x|)
    """
    X = np.asarray(X, dtype=np.float64).flatten()
    n = len(X)
    
    basis_list = []
    
    # Polynomial terms (degree 0 to 8)
    for d in range(9):
        basis_list.append(X ** d)
    
    # Trigonometric
    for k in [1, 2, 3]:
        basis_list.append(np.sin(k * X))
        basis_list.append(np.cos(k * X))
    
    # Exponential
    basis_list.append(np.exp(X))
    basis_list.append(np.exp(-X))
    basis_list.append(np.exp(0.5 * X))
    basis_list.append(np.exp(-0.5 * X))
    
    # Logarithmic (safe versions)
    basis_list.append(np.log1p(np.abs(X)))
    # log(1+x) only where x > -1, otherwise use log(1+|x|)
    log_pos = np.where(X > -1, np.log1p(X), np.log1p(np.abs(X)))
    basis_list.append(log_pos)
    
    # Tanh
    basis_list.append(np.tanh(X))
    basis_list.append(np.tanh(2 * X))
    
    # Absolute value powers
    basis_list.append(np.abs(X))
    basis_list.append(np.abs(X) ** 2)
    basis_list.append(np.abs(X) ** 3)
    
    # Sqrt of absolute value
    basis_list.append(np.sqrt(np.abs(X)))
    
    # Additional: x * sin(x), x * cos(x)
    basis_list.append(X * np.sin(X))
    basis_list.append(X * np.cos(X))
    
    # x * exp(-x^2) - Gaussian-like bump
    basis_list.append(X * np.exp(-X ** 2))
    
    # 1 / (1 + x^2) - Lorentzian-like
    basis_list.append(1.0 / (1.0 + X ** 2))
    
    basis = np.column_stack(basis_list)
    
    # Replace any non-finite values with 0
    basis = np.where(np.isfinite(basis), basis, 0.0)
    
    return basis


def _fit_ridge_rich_basis(X_train, y_train, alpha):
    """
    Fit ridge regression on the rich basis.
    Returns coefficients.
    """
    B = _rich_basis(X_train)
    # Add small regularization to diagonal
    n_features = B.shape[1]
    A = B.T @ B + alpha * np.eye(n_features)
    b = B.T @ y_train
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(B, y_train, rcond=None)[0]
    return coef


def _predict_rich(X_test, coef):
    """
    Predict using the rich basis and given coefficients.
    """
    B = _rich_basis(X_test)
    return B @ coef


def fit(X_train, y_train):
    global _t_start
    _t_start = time.time()
    
    X = np.asarray(X_train, dtype=np.float64).flatten()
    y = np.asarray(y_train, dtype=np.float64).flatten()
    
    n = len(X)
    
    # Standardize X for numerical stability
    X_mean = X.mean()
    X_std = X.std() + 1e-8
    Xn = (X - X_mean) / X_std
    
    # Standardize y
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    yn = (y - y_mean) / y_std
    
    # Try a range of regularization strengths
    # Use log-spaced alphas from small to large
    alphas = np.logspace(-3, 3, 25)
    
    best_alpha = 1.0
    best_cv_mse = np.inf
    best_coef = None
    
    # Simple K-fold cross-validation (5-fold) for alpha selection
    # Since N=200, 5-fold means 40 points per fold
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    for alpha in alphas:
        if _time_left() < 30:
            break
        
        fold_mses = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train_fold = Xn[train_idx]
            y_train_fold = yn[train_idx]
            X_val_fold = Xn[val_idx]
            y_val_fold = yn[val_idx]
            
            try:
                coef = _fit_ridge_rich_basis(X_train_fold, y_train_fold, alpha)
                y_pred = _predict_rich(X_val_fold, coef)
                mse = np.mean((y_pred - y_val_fold) ** 2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        cv_mse = np.mean(fold_mses)
        if cv_mse < best_cv_mse:
            best_cv_mse = cv_mse
            best_alpha = alpha
            # Save coefficients from this fold? No, we need to refit on all data later.
    
    # Refit on all training data with best alpha
    if _time_left() > 10:
        best_coef = _fit_ridge_rich_basis(Xn, yn, best_alpha)
    else:
        # Fallback: use a reasonable default
        best_alpha = 1.0
        best_coef = _fit_ridge_rich_basis(Xn, yn, best_alpha)
    
    # Final prediction function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64).flatten()
        # Standardize
        Xn_test = (X_test - X_mean) / X_std
        # Predict in standardized space
        y_pred_std = _predict_rich(Xn_test, best_coef)
        # Denormalize
        return y_pred_std * y_std + y_mean
    
    return predict
