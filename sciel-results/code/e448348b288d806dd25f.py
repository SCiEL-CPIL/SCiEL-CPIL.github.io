"""Symbolic regression with ensemble of Kernel Ridge Regressors and adaptive selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracker
_total_start = None
_time_limit = 315.0  # Leave 15s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _rbf_kernel(X1, X2, gamma):
    """RBF kernel: K(x, x') = exp(-gamma * ||x - x'||^2)"""
    sq1 = np.sum(X1 ** 2, axis=1)
    sq2 = np.sum(X2 ** 2, axis=1)
    dist_sq = sq1[:, None] + sq2[None, :] - 2 * X1 @ X2.T
    np.maximum(dist_sq, 0, out=dist_sq)
    return np.exp(-gamma * dist_sq)

def _poly_kernel(X1, X2, degree, coef0=1.0, gamma=1.0):
    """Polynomial kernel: K(x, x') = (gamma * x . x' + coef0)^degree"""
    return (gamma * X1 @ X2.T + coef0) ** degree

def _linear_kernel(X1, X2):
    """Linear kernel: K(x, x') = x . x'"""
    return X1 @ X2.T

def _fit_krr(X, y, kernel_func, alphas):
    """
    Fit Kernel Ridge Regressor with given kernel and regularization strengths.
    Returns list of (alpha, alpha_vec, kernel_func) for each alpha.
    """
    n = X.shape[0]
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    K = kernel_func(X, X)
    results = []
    
    for lam in alphas:
        A = K + lam * np.eye(n)
        try:
            alpha_vec = np.linalg.solve(A, y_norm)
        except np.linalg.LinAlgError:
            alpha_vec = np.linalg.lstsq(A, y_norm, rcond=None)[0]
        results.append((lam, alpha_vec))
    
    return results, y_mean, y_std

def _cv_score_krr(X, y, kernel_func, alphas, n_folds=5):
    """
    Cross-validate KRR with given kernel and regularization strengths.
    Returns best (cv_mse, lam, alpha_vec_full) using leave-one-out approximation
    or k-fold CV.
    """
    n = X.shape[0]
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Use k-fold CV
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    best_cv_mse = np.inf
    best_lam = alphas[0]
    best_alpha = None
    
    for lam in alphas:
        cv_errors = []
        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else n
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            
            X_train = X[train_idx]
            y_train = y_norm[train_idx]
            X_test = X[test_idx]
            y_test = y_norm[test_idx]
            
            K_train = kernel_func(X_train, X_train)
            K_test = kernel_func(X_test, X_train)
            
            A = K_train + lam * np.eye(len(train_idx))
            try:
                alpha_vec = np.linalg.solve(A, y_train)
            except np.linalg.LinAlgError:
                alpha_vec = np.linalg.lstsq(A, y_train, rcond=None)[0]
            
            pred = K_test @ alpha_vec
            mse = np.mean((pred - y_test) ** 2)
            cv_errors.append(mse)
        
        mean_cv_mse = np.mean(cv_errors)
        if mean_cv_mse < best_cv_mse:
            best_cv_mse = mean_cv_mse
            best_lam = lam
    
    # Refit on all data with best lambda
    K_full = kernel_func(X, X)
    A_full = K_full + best_lam * np.eye(n)
    try:
        best_alpha = np.linalg.solve(A_full, y_norm)
    except np.linalg.LinAlgError:
        best_alpha = np.linalg.lstsq(A_full, y_norm, rcond=None)[0]
    
    return best_cv_mse, best_lam, best_alpha

def _build_rich_features(X):
    """Build rich feature matrix for linear/poly kernel fallback."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones(n)]
    
    # Linear terms
    for j in range(d):
        cols.append(X[:, j])
    
    # Polynomial terms up to degree 3 (keep moderate to avoid explosion)
    for deg in range(2, 4):
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
    
    # Trigonometric terms
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
    
    # Products of pairs
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    
    return np.column_stack(cols)

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Normalize X for better kernel behavior
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-12] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Time check
    if _remaining() < 10:
        # Fallback: simple linear
        F = _build_rich_features(X_norm)
        F_mean_f = F.mean(axis=0)
        F_std_f = F.std(axis=0)
        F_std_f[F_std_f < 1e-12] = 1.0
        F_n = (F - F_mean_f) / F_std_f
        w, *_ = np.linalg.lstsq(F_n, y_norm, rcond=None)
        
        def predict(X_test):
            X_t = X_test.astype(np.float64)
            X_tn = (X_t - X_mean) / X_std
            F_t = _build_rich_features(X_tn)
            F_tn = (F_t - F_mean_f) / F_std_f
            return (F_tn @ w) * y_std + y_mean
        return predict
    
    # Define kernel configurations
    # RBF kernels with different gammas
    rbf_gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    # Polynomial kernels with different degrees
    poly_configs = [(2, 1.0, 0.5), (3, 1.0, 0.5), (2, 1.0, 1.0), (3, 1.0, 1.0)]
    # Regularization strengths
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    # Collect all kernel candidates
    kernel_candidates = []
    
    # RBF kernels
    for gamma in rbf_gammas:
        kf = lambda X1, X2, g=gamma: _rbf_kernel(X1, X2, g)
        kernel_candidates.append(('rbf', gamma, kf))
    
    # Polynomial kernels
    for deg, coef0, gam in poly_configs:
        kf = lambda X1, X2, dd=deg, cc=coef0, gg=gam: _poly_kernel(X1, X2, dd, cc, gg)
        kernel_candidates.append(('poly', (deg, coef0, gam), kf))
    
    # Linear kernel
    kernel_candidates.append(('linear', 1.0, _linear_kernel))
    
    # Cross-validate each kernel configuration
    best_overall_mse = np.inf
    best_kernel_name = None
    best_kernel_params = None
    best_kernel_func = None
    best_alpha_vec = None
    best_lam = None
    
    for kname, kparams, kf in kernel_candidates:
        if _remaining() < 15:
            break
        
        try:
            cv_mse, best_lam_k, alpha_vec = _cv_score_krr(X_norm, y_norm, kf, alphas, n_folds=5)
            
            if cv_mse < best_overall_mse:
                best_overall_mse = cv_mse
                best_kernel_name = kname
                best_kernel_params = kparams
                best_kernel_func = kf
                best_alpha_vec = alpha_vec
                best_lam = best_lam_k
        except Exception:
            continue
    
    # Also try rich feature linear regression as additional candidate
    try:
        F = _build_rich_features(X_norm)
        F_mean_f = F.mean(axis=0)
        F_std_f = F.std(axis=0)
        F_std_f[F_std_f < 1e-12] = 1.0
        F_n = (F - F_mean_f) / F_std_f
        
        # Simple CV for linear model
        indices = np.arange(n)
        np.random.seed(43)
        np.random.shuffle(indices)
        fold_size = n // 5
        cv_errors = []
        for fold in range(5):
            start = fold * fold_size
            end = start + fold_size if fold < 4 else n
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            
            F_train = F_n[train_idx]
            y_train_f = y_norm[train_idx]
            F_test = F_n[test_idx]
            y_test_f = y_norm[test_idx]
            
            w_cv, *_ = np.linalg.lstsq(F_train, y_train_f, rcond=None)
            pred = F_test @ w_cv
            cv_errors.append(np.mean((pred - y_test_f) ** 2))
        
        lin_cv_mse = np.mean(cv_errors)
        
        if lin_cv_mse < best_overall_mse:
            best_overall_mse = lin_cv_mse
            best_kernel_name = 'linear_features'
            best_kernel_params = None
            best_kernel_func = None
            best_alpha_vec = None
            best_lam = None
            # Store the linear model weights
            w_full, *_ = np.linalg.lstsq(F_n, y_norm, rcond=None)
            best_linear_w = w_full
            best_F_mean = F_mean_f
            best_F_std = F_std_f
    except Exception:
        pass
    
    # If we ended up with linear features model
    if best_kernel_name == 'linear_features':
        def predict(X_test):
            X_t = X_test.astype(np.float64)
            X_tn = (X_t - X_mean) / X_std
            F_t = _build_rich_features(X_tn)
            F_tn = (F_t - best_F_mean) / best_F_std
            return (F_tn @ best_linear_w) * y_std + y_mean
        return predict
    
    # Build prediction function using the best kernel model
    if best_kernel_func is not None:
        K_train_full = best_kernel_func(X_norm, X_norm)
        # We already have best_alpha_vec from CV refit
        
        def predict(X_test):
            X_t = X_test.astype(np.float64)
            X_tn = (X_t - X_mean) / X_std
            K_test = best_kernel_func(X_tn, X_norm)
            y_pred_norm = K_test @ best_alpha_vec
            return y_pred_norm * y_std + y_mean
        return predict
    
    # Ultimate fallback: constant prediction
    def predict(X_test):
        return np.full(X_test.shape[0], y_mean)
    return predict
# EVOLVE-BLOCK-END
