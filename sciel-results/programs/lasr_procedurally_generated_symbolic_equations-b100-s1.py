"""Symbolic regression via ensemble of diverse models with time budget guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

# Global deadline: 330s wall-clock, use 300s as safety margin
_DEADLINE = time.time() + 300.0


def _remaining():
    return _DEADLINE - time.time()


def _poly_fit(X, y, degree):
    """Polynomial fit of given degree."""
    n = len(X)
    # Build Vandermonde-like feature matrix for 1D case
    x = X[:, 0]
    cols = [x ** k for k in range(degree + 1)]
    F = np.column_stack(cols) if degree > 0 else np.ones((n, 1))
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    return lambda X: (np.column_stack([X[:, 0] ** k for k in range(degree + 1)]) if degree > 0 else np.ones((len(X), 1))) @ w


def _ridge_fit(X, y, alpha=1e-3):
    """Ridge regression on a rich basis: polynomials, Fourier, exponentials, sigmoids."""
    x = X[:, 0]
    n = len(x)
    
    # Build rich feature matrix
    cols = [np.ones_like(x)]
    # Polynomials up to degree 8
    for k in range(1, 9):
        cols.append(x ** k)
    # Fourier terms
    for k in range(1, 8):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    # Exponential-like terms (bounded)
    for a in [0.5, 1.0, 2.0]:
        cols.append(np.tanh(a * x))
        cols.append(np.exp(-a * np.abs(x)))
    # Reciprocal-like (safe)
    cols.append(1.0 / (1.0 + np.abs(x) + 1e-6))
    cols.append(x / (1.0 + np.abs(x)))
    
    F = np.column_stack(cols)
    # Ridge solve: (F^T F + alpha*I) w = F^T y
    FTF = F.T @ F + alpha * np.eye(F.shape[1])
    FTy = F.T @ y
    w = np.linalg.solve(FTF, FTy)
    
    def predict(X_new):
        x_new = X_new[:, 0]
        cols = [np.ones_like(x_new)]
        for k in range(1, 9):
            cols.append(x_new ** k)
        for k in range(1, 8):
            cols.append(np.sin(k * x_new))
            cols.append(np.cos(k * x_new))
        for a in [0.5, 1.0, 2.0]:
            cols.append(np.tanh(a * x_new))
            cols.append(np.exp(-a * np.abs(x_new)))
        cols.append(1.0 / (1.0 + np.abs(x_new) + 1e-6))
        cols.append(x_new / (1.0 + np.abs(x_new)))
        F_new = np.column_stack(cols)
        return F_new @ w
    return predict


def _spline_fit(X, y, n_knots=50):
    """Interpolating spline through sorted data points with limited knots."""
    x = X[:, 0]
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    # Use a subset of points as knots to avoid overfitting
    if len(x_sorted) > n_knots:
        idx = np.linspace(0, len(x_sorted) - 1, n_knots, dtype=int)
        x_knots = x_sorted[idx]
        y_knots = y_sorted[idx]
    else:
        x_knots = x_sorted
        y_knots = y_sorted
    
    # Use cubic interpolation if enough knots, else linear
    if len(x_knots) >= 4:
        f = interp1d(x_knots, y_knots, kind='cubic', fill_value='extrapolate')
    else:
        f = interp1d(x_knots, y_knots, kind='linear', fill_value='extrapolate')
    return lambda X: f(X[:, 0])


def _gaussian_process_fit(X, y, length_scale, variance, nu=2.5):
    """GP with Matérn kernel (nu=2.5 for smoothness)."""
    x = X[:, 0]
    n = len(x)
    
    def kernel(a, b):
        d = np.abs(a[:, None] - b[None, :])
        # Matérn-5/2 kernel
        sqrt3d = np.sqrt(3.0) * d / length_scale
        return variance * (1.0 + sqrt3d + sqrt3d**2 / 3.0) * np.exp(-sqrt3d)
    
    K = kernel(x, x) + 1e-6 * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        # Fallback with more jitter
        K2 = kernel(x, x) + 1e-4 * np.eye(n)
        try:
            L = np.linalg.cholesky(K2)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        except np.linalg.LinAlgError:
            return None
    
    def predict(X_new):
        x_new = X_new[:, 0]
        K_star = kernel(x_new, x)
        return K_star @ alpha
    return predict


def _cross_validate_gp(X, y, n_folds=5):
    """Cross-validate GP hyperparameters and return best predict function."""
    x = X[:, 0]
    n = len(x)
    var_y = np.var(y) + 1e-10
    x_range = np.ptp(x)
    
    # Grid of hyperparameters to try
    length_scales = [x_range * f for f in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]]
    variances = [var_y * f for f in [0.5, 1.0, 2.0]]
    
    best_cv_mse = np.inf
    best_params = None
    
    # Shuffle for CV
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    fold_size = n // n_folds
    
    for ls in length_scales:
        if _remaining() < 20:
            break
        for var in variances:
            cv_mses = []
            for fold in range(n_folds):
                val_idx = perm[fold * fold_size:(fold + 1) * fold_size]
                train_idx = np.concatenate([perm[:fold * fold_size], perm[(fold + 1) * fold_size:]])
                
                X_tr = X[train_idx]
                y_tr = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                try:
                    pred_fn = _gaussian_process_fit(X_tr, y_tr, ls, var)
                    if pred_fn is None:
                        cv_mses.append(np.inf)
                        continue
                    pred = pred_fn(X_val)
                    mse = np.mean((pred - y_val) ** 2)
                    cv_mses.append(mse)
                except Exception:
                    cv_mses.append(np.inf)
            
            mean_cv_mse = np.mean(cv_mses)
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_params = (ls, var)
    
    if best_params is None:
        return None
    
    # Fit on all data with best hyperparameters
    return _gaussian_process_fit(X, y, best_params[0], best_params[1])


def _neural_net_fit(X, y, hidden_size, n_epochs=500, lr=0.01):
    """Simple MLP with one hidden layer, trained via gradient descent with adaptive learning rate."""
    x = X[:, 0].reshape(-1, 1)
    y_col = y.reshape(-1, 1)
    n = len(x)
    
    # Initialize weights
    np.random.seed(42)
    W1 = np.random.randn(1, hidden_size) * np.sqrt(2.0 / 1)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, 1))
    
    best_loss = np.inf
    best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
    patience = 0
    max_patience = 100
    
    for epoch in range(n_epochs):
        if _remaining() < 5:
            break
        # Forward
        z1 = x @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        pred = z2
        
        # Backward
        dz2 = (2.0 / n) * (pred - y_col)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - a1 ** 2)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        # Track best
        loss = np.mean((pred - y_col) ** 2)
        if loss < best_loss - 1e-8:
            best_loss = loss
            best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
    
    W1, b1, W2, b2 = best_W1, best_b1, best_W2, best_b2
    
    def predict(X_new):
        x_new = X_new[:, 0].reshape(-1, 1)
        z1 = x_new @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        return z2.flatten()
    return predict


def _fourier_fit(X, y, n_terms):
    """Fourier series fit."""
    x = X[:, 0]
    cols = [np.ones_like(x)]
    for k in range(1, n_terms + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    F = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    
    def predict(X_new):
        x_new = X_new[:, 0]
        cols = [np.ones_like(x_new)]
        for k in range(1, n_terms + 1):
            cols.append(np.sin(k * x_new))
            cols.append(np.cos(k * x_new))
        F_new = np.column_stack(cols)
        return F_new @ w
    return predict


def _blend_predict(predictors, weights):
    """Blend multiple predictors with given weights."""
    def predict(X_new):
        preds = np.column_stack([p(X_new) for p, w in zip(predictors, weights)])
        return preds @ weights
    return predict


def fit(X_train, y_train):
    """Fit multiple models and return the best one based on training MSE."""
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    candidates = []
    
    # 1. Polynomial fits of various degrees
    for deg in [1, 2, 3, 4, 5, 6, 8, 10]:
        if _remaining() < 10:
            break
        try:
            pred_fn = _poly_fit(X, y, deg)
            train_pred = pred_fn(X)
            mse = np.mean((train_pred - y) ** 2)
            candidates.append((mse, pred_fn))
        except Exception:
            pass
    
    # 2. Spline interpolation with different knot counts
    for n_knots in [20, 50, 100]:
        if _remaining() < 10:
            break
        try:
            pred_fn = _spline_fit(X, y, n_knots)
            train_pred = pred_fn(X)
            mse = np.mean((train_pred - y) ** 2)
            candidates.append((mse, pred_fn))
        except Exception:
            pass
    
    # 3. Fourier series
    for n_terms in [3, 5, 8, 12, 20]:
        if _remaining() < 10:
            break
        try:
            pred_fn = _fourier_fit(X, y, n_terms)
            train_pred = pred_fn(X)
            mse = np.mean((train_pred - y) ** 2)
            candidates.append((mse, pred_fn))
        except Exception:
            pass
    
    # 4. Ridge regression with rich basis
    if _remaining() > 10:
        for alpha in [1e-4, 1e-3, 1e-2]:
            try:
                pred_fn = _ridge_fit(X, y, alpha)
                train_pred = pred_fn(X)
                mse = np.mean((train_pred - y) ** 2)
                candidates.append((mse, pred_fn))
            except Exception:
                pass
    
    # 5. Neural network
    if _remaining() > 20:
        for hidden in [8, 16, 32]:
            if _remaining() < 15:
                break
            try:
                pred_fn = _neural_net_fit(X, y, hidden, n_epochs=300, lr=0.01)
                train_pred = pred_fn(X)
                mse = np.mean((train_pred - y) ** 2)
                candidates.append((mse, pred_fn))
            except Exception:
                pass
    
    # 6. Gaussian Process with cross-validated hyperparameters
    if _remaining() > 30:
        try:
            pred_fn = _cross_validate_gp(X, y, n_folds=5)
            if pred_fn is not None:
                train_pred = pred_fn(X)
                mse = np.mean((train_pred - y) ** 2)
                candidates.append((mse, pred_fn))
        except Exception:
            pass
    
    # 7. GP with fixed hyperparameters as backup
    if _remaining() > 15:
        var_y = np.var(y) + 1e-10
        x_range = np.ptp(X[:, 0])
        for ls_factor in [0.05, 0.1, 0.2, 0.5]:
            if _remaining() < 10:
                break
            try:
                pred_fn = _gaussian_process_fit(X, y, x_range * ls_factor, var_y)
                if pred_fn is not None:
                    train_pred = pred_fn(X)
                    mse = np.mean((train_pred - y) ** 2)
                    candidates.append((mse, pred_fn))
            except Exception:
                pass
    
    # Select best model by training MSE
    if not candidates:
        # Fallback: linear fit
        pred_fn = _poly_fit(X, y, 1)
        return pred_fn
    
    candidates.sort(key=lambda t: t[0])
    best_mse, best_pred = candidates[0]
    
    # Try blending top 3 candidates if they're close in performance
    if len(candidates) >= 3 and _remaining() > 10:
        top3 = candidates[:3]
        mses = np.array([m for m, _ in top3])
        preds = [p for _, p in top3]
        
        # Check if top models are within 2x of best
        if mses[2] < 2.0 * mses[0] + 1e-10:
            # Inverse-MSE weighting
            inv_mses = 1.0 / (mses + 1e-10)
            weights = inv_mses / inv_mses.sum()
            blend_fn = _blend_predict(preds, weights)
            blend_pred = blend_fn(X)
            blend_mse = np.mean((blend_pred - y) ** 2)
            
            if blend_mse < best_mse:
                best_pred = blend_fn
                best_mse = blend_mse
    
    # Final safety check
    def safe_predict(X_new):
        try:
            result = best_pred(X_new)
            result = np.asarray(result, dtype=np.float64).flatten()
            # Ensure finite values
            if not np.all(np.isfinite(result)):
                result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            return result
        except Exception:
            return np.zeros(len(X_new))
    
    return safe_predict
# EVOLVE-BLOCK-END
