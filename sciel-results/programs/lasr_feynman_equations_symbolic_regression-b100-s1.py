"""Symbolic regression with rich feature expansion and adaptive selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
import itertools

# Global time tracker
_total_start = None
_time_limit = 320.0  # Leave 10s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    # Build rich feature matrix
    def build_features(X):
        n = X.shape[0]
        d = X.shape[1]
        cols = [np.ones(n)]
        
        # Linear terms
        for j in range(d):
            cols.append(X[:, j])
        
        # Polynomial terms up to degree 4
        for deg in range(2, 5):
            for combo in range(d):
                # Generate all monomials of degree `deg` with d variables
                _poly_monomials = []
                def _gen(idx, remaining, current):
                    if idx == d - 1:
                        _poly_monomials.append(current + [remaining])
                        return
                    for k in range(remaining + 1):
                        _gen(idx + 1, remaining - k, current + [k])
                _gen(0, deg, [])
                for exps in _poly_monomials:
                    term = np.ones(n)
                    for j in range(d):
                        if exps[j] > 0:
                            term = term * np.power(X[:, j], exps[j])
                    cols.append(term)
        
        # Trigonometric terms: sin, cos of each input
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
        
        # sin/cos of sums of pairs
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(X[:, i] + X[:, j]))
                cols.append(np.cos(X[:, i] + X[:, j]))
                cols.append(np.sin(X[:, i] - X[:, j]))
                cols.append(np.cos(X[:, i] - X[:, j]))
        
        # Exponential and log terms
        for j in range(d):
            cols.append(np.exp(X[:, j] / 5.0))  # scaled to avoid overflow
            cols.append(np.log(X[:, j] + 1.0))
        
        # Products of pairs
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
                cols.append(np.sqrt(np.maximum(X[:, i] * X[:, j], 0)))
        
        # Ratio terms (safe)
        for i in range(d):
            for j in range(i + 1, d):
                denom = X[:, j] + 1e-10
                cols.append(X[:, i] / denom)
        
        return np.column_stack(cols)
    
    F = build_features(X)
    n_features = F.shape[1]
    
    # Normalize features for better conditioning
    F_mean = F.mean(axis=0)
    F_std = F.std(axis=0)
    F_std[F_std < 1e-10] = 1.0
    F_norm = (F - F_mean) / F_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Try Lasso-style feature selection using iterative hard thresholding
    # Start with ridge, then prune
    
    def ridge_solve(F_mat, y_vec, lam):
        n_feat = F_mat.shape[1]
        # Add regularization
        A = F_mat.T @ F_mat + lam * np.eye(n_feat)
        b = F_mat.T @ y_vec
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        return w
    
    # Try several lambda values for ridge, pick best via cross-validation-like approach
    # But with 200 points, we can do a simple train/test split
    # Actually, let's try elastic net via scipy if available, otherwise manual
    
    # Try different lambda values with cross-validation
    best_score = np.inf
    best_w = None
    best_lam = 0.0
    
    # Use a subset of lambdas
    lam_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    # Simple 5-fold cross-validation
    fold_size = n // 5
    folds = []
    for i in range(5):
        start = i * fold_size
        end = start + fold_size if i < 4 else n
        folds.append((start, end))
    
    for lam in lam_values:
        cv_scores = []
        for fold_idx, (start, end) in enumerate(folds):
            # Create validation and training indices
            val_idx = np.arange(start, end)
            train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])
            
            F_train = F_norm[train_idx]
            y_train_fold = y_norm[train_idx]
            F_val = F_norm[val_idx]
            y_val = y_norm[val_idx]
            
            w = ridge_solve(F_train, y_train_fold, lam)
            pred_val = F_val @ w
            mse_val = np.mean((pred_val - y_val) ** 2)
            cv_scores.append(mse_val)
        
        mean_cv = np.mean(cv_scores)
        if mean_cv < best_score:
            best_score = mean_cv
            best_lam = lam
        
        # Check time
        if _remaining() < 5.0:
            break
    
    # Fit final model on all data with best lambda
    best_w = ridge_solve(F_norm, y_norm, best_lam)
    best_score = np.mean((F_norm @ best_w - y_norm) ** 2)
    
    # Now try L1-style pruning: start from ridge solution, zero out small coefficients
    # and refit with more aggressive thresholding
    w_current = best_w.copy()
    for iteration in range(10):
        # Check time
        if _remaining() < 10.0:
            break
        
        # Identify active features (non-zero coefficients)
        # Use adaptive threshold based on coefficient distribution
        coeff_abs = np.abs(w_current)
        threshold = max(0.001, np.percentile(coeff_abs[coeff_abs > 0], 20))
        mask = coeff_abs > threshold
        if not np.any(mask) or np.sum(mask) < 3:
            break
        
        F_active = F_norm[:, mask]
        # Refit on active features only
        try:
            w_active, *_ = np.linalg.lstsq(F_active, y_norm, rcond=None)
        except:
            break
        # Build full solution
        w_new = np.zeros_like(w_current)
        w_new[mask] = w_active
        # Check if solution improved
        pred_new = F_norm @ w_new
        mse_new = np.mean((pred_new - y_norm) ** 2)
        if mse_new < best_score * 1.005:
            best_score = mse_new
            best_w = w_new
            w_current = w_new
        else:
            break
    
    # Final: also try pure least squares on all features as fallback
    # And try a few more regularization strengths around the best
    try:
        w_ls, *_ = np.linalg.lstsq(F_norm, y_norm, rcond=None)
        pred_ls = F_norm @ w_ls
        mse_ls = np.mean((pred_ls - y_norm) ** 2)
        if mse_ls < best_score:
            best_w = w_ls
            best_score = mse_ls
    except:
        pass
    
    # Try nearby lambda values for fine-tuning
    if _remaining() > 15.0:
        nearby_lams = [best_lam * 0.5, best_lam * 2.0, best_lam * 0.1, best_lam * 10.0]
        for lam_nearby in nearby_lams:
            if lam_nearby <= 0:
                continue
            w_nearby = ridge_solve(F_norm, y_norm, lam_nearby)
            pred_nearby = F_norm @ w_nearby
            mse_nearby = np.mean((pred_nearby - y_norm) ** 2)
            if mse_nearby < best_score:
                best_score = mse_nearby
                best_w = w_nearby
    
    # Build prediction function
    def predict(X_test):
        X_t = X_test.astype(np.float64)
        F_t = build_features(X_t)
        F_t_norm = (F_t - F_mean) / F_std
        y_pred = F_t_norm @ best_w
        # Ensure finite predictions
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        return y_pred * y_std + y_mean
    
    return predict
# EVOLVE-BLOCK-END
