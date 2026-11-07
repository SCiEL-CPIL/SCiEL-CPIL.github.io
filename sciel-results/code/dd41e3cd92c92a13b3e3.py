"""Symbolic regression over 10 hidden equations: Nyström sparse kernel ensemble + exact basis search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.linalg

_START = time.time()
_DEADLINE = 300.0  # safety margin vs 330s kill

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _make_predict_rbf(X_train, y_train, gamma, alpha):
    """Create a predict function for RBF Kernel Ridge Regression."""
    diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    K = np.exp(-gamma * sq_dists)
    K_reg = K + alpha * np.eye(len(X_train))
    try:
        w = scipy.linalg.solve(K_reg, y_train)
    except Exception:
        w = np.linalg.pinv(K_reg) @ y_train
        
    def predict(X):
        diff_test = X[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        sq_dists_test = np.sum(diff_test**2, axis=-1)
        K_test = np.exp(-gamma * sq_dists_test)
        return K_test @ w
        
    return predict

def _make_predict_linear(basis_func, w):
    """Create a predict function for linear combination of basis features."""
    def predict(X):
        A = basis_func(X)
        return A @ w
    return predict

def _try_exact_fit(basis_func, X, y, tol=1e-6):
    """Try to fit y using linear combination of basis_func(X)."""
    try:
        A = basis_func(X)
        if not np.all(np.isfinite(A)):
            return None
        w, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        if not np.all(np.isfinite(pred)):
            return None
        mse = np.mean((pred - y)**2)
        scale = max(1.0, np.max(np.abs(y)))
        if mse <= (tol * scale)**2:
            return w, mse
    except Exception:
        pass
    return None

# --- Basis Function Generators ---

def _basis_poly2(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)

def _basis_poly3(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    for i in range(d):
        cols.append(X[:, i] ** 3)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _basis_poly4(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
        cols.append(X[:, j] ** 4)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] ** 2 * X[:, j])
            cols.append(X[:, i] * X[:, j] ** 2)
            cols.append(X[:, i] ** 2 * X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _basis_rich(X):
    """Includes polynomials, transcendentals, and common functions."""
    d = X.shape[1]
    cols = [np.ones(len(X))]
    
    # Polynomials up to degree 2
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            
    # Transcendentals and common functions on individual variables
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.log(x))
        cols.append(np.sqrt(x))
        cols.append(np.abs(x))
        cols.append(x ** 0.5)
        cols.append(np.exp(-x))
        cols.append(np.sin(x * x))
        cols.append(np.cos(x * x))
        
    # Some key interactions
    if d >= 2:
        for i in range(d):
            for j in range(i+1, d):
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.tanh(X[:, i]) * X[:, j])
                cols.append(np.log(X[:, i]) * X[:, j])
                
    return np.column_stack(cols)

def _basis_trig_poly(X):
    """Polynomials and trigonometric functions with more interactions."""
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
        cols.append(np.tanh(X[:, j]))
        cols.append(np.arctan(X[:, j]))
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
            cols.append(np.sin(X[:, i] + X[:, j]))
            cols.append(np.cos(X[:, i] + X[:, j]))
    return np.column_stack(cols)

def _basis_comprehensive(X):
    """Very comprehensive basis for complex functions."""
    d = X.shape[1]
    cols = [np.ones(len(X))]
    
    # All polynomial terms up to degree 3
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] ** 2 * X[:, j])
            cols.append(X[:, i] * X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    
    # Transcendentals
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.log(x))
        cols.append(np.sqrt(x))
        cols.append(np.exp(-x))
        cols.append(np.sin(x**2))
        cols.append(np.cos(x**2))
        cols.append(np.tanh(x**2))
        
    # Interactions with transcendentals
    if d >= 2:
        for i in range(d):
            for j in range(i+1, min(i+2, d)):
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.tanh(X[:, i]) * X[:, j])
                cols.append(np.log(X[:, i]) * X[:, j])
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(X[:, i] * np.tanh(X[:, j]))
                
    return np.column_stack(cols)

_CANDIDATE_BASES = [
    _basis_poly2,
    _basis_poly3,
    _basis_poly4,
    _basis_rich,
    _basis_trig_poly,
    _basis_comprehensive
]

def _cv_rbf_score(X, y, gamma, alpha, n_folds=5):
    """Cross-validate RBF parameters."""
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    cv_errors = []
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        if len(train_idx) < 10:
            continue
            
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        try:
            pred = _make_predict_rbf(X_train_fold, y_train_fold, gamma, alpha)(X_val_fold)
            err = np.mean((pred - y_val_fold)**2)
            if np.isfinite(err):
                cv_errors.append(err)
        except Exception:
            continue
    
    if cv_errors:
        return np.mean(cv_errors)
    return float('inf')

def _make_predict_poly_kernel(X_train, y_train, degree, gamma, alpha):
    """Create a predict function for Polynomial Kernel Ridge Regression."""
    def _poly_kernel(A, B, degree, gamma):
        return (gamma * (A @ B.T) + 1) ** degree
    
    K = _poly_kernel(X_train, X_train, degree, gamma)
    K_reg = K + alpha * np.eye(len(X_train))
    try:
        w = scipy.linalg.solve(K_reg, y_train)
    except Exception:
        w = np.linalg.pinv(K_reg) @ y_train
        
    def predict(X):
        K_test = _poly_kernel(X, X_train, degree, gamma)
        return K_test @ w
        
    return predict

def _cv_poly_kernel_score(X, y, degree, gamma, alpha, n_folds=5):
    """Cross-validate polynomial kernel parameters."""
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    cv_errors = []
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        if len(train_idx) < 10:
            continue
            
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        try:
            pred = _make_predict_poly_kernel(X_train_fold, y_train_fold, degree, gamma, alpha)(X_val_fold)
            err = np.mean((pred - y_val_fold)**2)
            if np.isfinite(err):
                cv_errors.append(err)
        except Exception:
            continue
    
    if cv_errors:
        return np.mean(cv_errors)
    return float('inf')

# --- Nyström Approximation ---

def _nystrom_rbf_predict(X_train, y_train, gamma, alpha, n_landmarks):
    """Create a predict function using Nyström approximation of RBF kernel."""
    n = len(X_train)
    
    # Select landmarks using random sampling
    np.random.seed(42)
    landmark_idx = np.random.choice(n, min(n_landmarks, n), replace=False)
    X_land = X_train[landmark_idx]
    y_land = y_train[landmark_idx]
    
    # Compute K_l: landmarks to all training points
    diff_l = X_land[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    sq_dists_l = np.sum(diff_l**2, axis=-1)
    K_l = np.exp(-gamma * sq_dists_l)  # shape (m, n)
    
    # Compute K_ll: landmarks to landmarks
    diff_ll = X_land[:, np.newaxis, :] - X_land[np.newaxis, :, :]
    sq_dists_ll = np.sum(diff_ll**2, axis=-1)
    K_ll = np.exp(-gamma * sq_dists_ll)  # shape (m, m)
    
    m = K_ll.shape[0]
    K_ll_reg = K_ll + alpha * np.eye(m)
    
    try:
        # Solve for weights in landmark space
        # We want: K @ w = y, where K is approximated as K_l^T @ K_ll^{-1} @ K_l
        # So w = K_l^T @ (K_ll + alpha I)^{-1} @ K_l @ ... 
        # Actually for kernel ridge regression with Nyström:
        # f(x) = K(x, X_land) @ K_ll^{-1} @ (K_ll + alpha I)^{-1} @ K_l @ y
        # Let's define: v = (K_ll + alpha I)^{-1} @ K_l @ y
        # Then f(x) = K(x, X_land) @ v
        Kly = K_l @ y_train
        v = scipy.linalg.solve(K_ll_reg, Kly)
    except Exception:
        v = np.linalg.pinv(K_ll_reg) @ Kly
    
    def predict(X):
        diff_test = X[:, np.newaxis, :] - X_land[np.newaxis, :, :]
        sq_dists_test = np.sum(diff_test**2, axis=-1)
        K_test_land = np.exp(-gamma * sq_dists_test)  # shape (n_test, m)
        return K_test_land @ v
    
    return predict

def _cv_nystrom_rbf_score(X, y, gamma, alpha, n_landmarks, n_folds=5):
    """Cross-validate Nyström RBF parameters."""
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    cv_errors = []
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        if len(train_idx) < 10:
            continue
            
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        try:
            pred = _nystrom_rbf_predict(X_train_fold, y_train_fold, gamma, alpha, n_landmarks)(X_val_fold)
            err = np.mean((pred - y_val_fold)**2)
            if np.isfinite(err):
                cv_errors.append(err)
        except Exception:
            continue
    
    if cv_errors:
        return np.mean(cv_errors)
    return float('inf')

def _make_predict_ensemble(predictors, weights=None):
    """Create an ensemble predict function from multiple predictors."""
    if weights is None:
        weights = np.ones(len(predictors)) / len(predictors)
    
    def predict(X):
        preds = np.array([p(X) for p in predictors])
        return np.sum(preds * weights[:, np.newaxis], axis=0)
    
    return predict

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    
    n_samples = len(X)
    d = X.shape[1]
    
    # 1. Try Exact Fit with various bases
    best_exact_predict = None
    best_exact_err = float('inf')
    
    for basis_func in _CANDIDATE_BASES:
        if _remaining() < 15.0:
            break
        try:
            result = _try_exact_fit(basis_func, X, y)
            if result is not None:
                w, err = result
                if err < best_exact_err:
                    best_exact_err = err
                    best_exact_predict = _make_predict_linear(basis_func, w)
        except Exception:
            continue
            
    if best_exact_predict is not None:
        return best_exact_predict
        
    # 2. Fallback: Kernel Ridge Regression with CV (RBF + Polynomial + Nyström)
    best_kernel_predict = None
    best_kernel_cv_err = float('inf')
    candidate_models = []  # (cv_err, predict_func)
    
    # RBF parameters - wider range
    gammas_rbf = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    for gamma in gammas_rbf:
        if _remaining() < 10.0:
            break
        for alpha in alphas:
            if _remaining() < 5.0:
                break
            try:
                cv_err = _cv_rbf_score(X, y, gamma, alpha)
                if cv_err < best_kernel_cv_err:
                    best_kernel_cv_err = cv_err
                    best_kernel_predict = _make_predict_rbf(X, y, gamma, alpha)
                if np.isfinite(cv_err):
                    candidate_models.append((cv_err, _make_predict_rbf(X, y, gamma, alpha)))
            except Exception:
                continue
                
    # Polynomial kernel parameters
    degrees = [2, 3, 4]
    gammas_poly = [0.1, 0.5, 1.0]
    
    for degree in degrees:
        if _remaining() < 10.0:
            break
        for gamma in gammas_poly:
            if _remaining() < 5.0:
                break
            for alpha in alphas:
                if _remaining() < 5.0:
                    break
                try:
                    cv_err = _cv_poly_kernel_score(X, y, degree, gamma, alpha)
                    if cv_err < best_kernel_cv_err:
                        best_kernel_cv_err = cv_err
                        best_kernel_predict = _make_predict_poly_kernel(X, y, degree, gamma, alpha)
                    if np.isfinite(cv_err):
                        candidate_models.append((cv_err, _make_predict_poly_kernel(X, y, degree, gamma, alpha)))
                except Exception:
                    continue
    
    # Nyström RBF parameters
    n_landmarks_list = [20, 50, 100]
    for gamma in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        if _remaining() < 10.0:
            break
        for alpha in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
            if _remaining() < 5.0:
                break
            for n_land in n_landmarks_list:
                if _remaining() < 3.0:
                    break
                try:
                    cv_err = _cv_nystrom_rbf_score(X, y, gamma, alpha, n_land)
                    if cv_err < best_kernel_cv_err:
                        best_kernel_cv_err = cv_err
                        best_kernel_predict = _nystrom_rbf_predict(X, y, gamma, alpha, n_land)
                    if np.isfinite(cv_err):
                        candidate_models.append((cv_err, _nystrom_rbf_predict(X, y, gamma, alpha, n_land)))
                except Exception:
                    continue
    
    if best_kernel_predict is not None:
        # Try ensemble of top 3 models if we have time
        if len(candidate_models) >= 3 and _remaining() > 20.0:
            candidate_models.sort(key=lambda x: x[0])
            top_models = [m[1] for m in candidate_models[:3]]
            top_errs = [m[0] for m in candidate_models[:3]]
            # Weight inversely proportional to error
            inv_errs = [1.0 / (e + 1e-10) for e in top_errs]
            total = sum(inv_errs)
            weights = [e / total for e in inv_errs]
            ensemble_predict = _make_predict_ensemble(top_models, weights)
            # Quick validation of ensemble
            try:
                # Use a simple holdout to check if ensemble is better
                n = len(y)
                val_idx = np.arange(n - 40, n)
                train_idx = np.arange(0, n - 40)
                pred_ens = ensemble_predict(X[val_idx])
                err_ens = np.mean((pred_ens - y[val_idx])**2)
                pred_best = best_kernel_predict(X[val_idx])
                err_best = np.mean((pred_best - y[val_idx])**2)
                if err_ens < err_best:
                    return ensemble_predict
            except Exception:
                pass
        return best_kernel_predict
        
    # 3. Ultimate Fallback: Ridge regression on comprehensive basis
    try:
        A = _basis_comprehensive(X)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return _make_predict_linear(_basis_comprehensive, w)
    except Exception:
        try:
            A = _basis_poly2(X)
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            return _make_predict_linear(_basis_poly2, w)
        except Exception:
            mean_y = np.mean(y)
            return lambda X: np.full(len(X), mean_y)

# EVOLVE-BLOCK-END
