"""Symbolic regression: RBF + Polynomial ensemble with cross-validation model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Budget: evaluator kills at 330s. We use 300s as our internal budget to be safe.
_TOTAL_BUDGET = 300.0
_GLOBAL_START = time.time()


def _remaining():
    return _TOTAL_BUDGET - (time.time() - _GLOBAL_START)


def _cv_score(models, X, y, k=5, seed=42):
    """
    Calculate mean CV MSE for a list of model-fitting functions.
    models: list of tuples (name, fit_func) where fit_func(X_tr, y_tr) returns predict_func
    """
    n = len(X)
    if n < k + 2:
        k = max(2, n - 2)
    
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    
    total_err = 0.0
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te, y_te = X[f], y[f]
        
        fold_errs = []
        for name, fit_func in models:
            try:
                if _remaining() < 5:
                    return np.inf
                predict_func = fit_func(X_tr, y_tr)
                preds = predict_func(X_te)
                preds = np.asarray(preds, dtype=np.float64).flatten()
                if len(preds) != len(y_te):
                    continue
                if not np.all(np.isfinite(preds)):
                    continue
                err = np.mean((preds - y_te) ** 2)
                fold_errs.append(err)
            except Exception:
                continue
        
        if fold_errs:
            # Take the best model's error for this fold? Or average?
            # For model selection, we want the error of the specific model.
            # This function returns a single score.
            # Let's return the mean of the best performing model across folds?
            # No, standard CV for a *specific* model configuration.
            # The caller should pass a list of *single* model configs to compare?
            # Or we can return a dict of mean errors per model name.
            pass
            
    # Actually, let's restructure. We want to compare different hyperparameters.
    # Let's just compute the mean CV error for a *single* model configuration.
    # The caller will loop over configurations.
    raise NotImplementedError("Use _cv_single_model instead")


def _cv_single_model(fit_func, X, y, k=5, seed=42):
    """
    Calculate mean CV MSE for a single model-fitting function.
    """
    n = len(X)
    if n < k + 2:
        k = max(2, n - 2)
    
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    
    errs = []
    for f in folds:
        if _remaining() < 5:
            return np.inf
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te, y_te = X[f], y[f]
        
        try:
            predict_func = fit_func(X_tr, y_tr)
            preds = predict_func(X_te)
            preds = np.asarray(preds, dtype=np.float64).flatten()
            if len(preds) != len(y_te):
                continue
            if not np.all(np.isfinite(preds)):
                continue
            err = np.mean((preds - y_te) ** 2)
            errs.append(err)
        except Exception:
            return np.inf
            
    if not errs:
        return np.inf
    return np.mean(errs)


def _fit_rbf(X, y, sigma):
    """
    Fit Gaussian RBF model.
    Returns a predict function.
    """
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    
    # Precompute kernel matrix
    # K[i, j] = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    # For 1D: K[i, j] = exp(-(x_i - x_j)^2 / (2 * sigma^2))
    
    # Use broadcasting for speed
    diff = x[:, None] - x[None, :]
    K = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Solve (K + lambda*I) @ w = y
    # Add small regularization for stability
    reg = 1e-6
    K_reg = K + reg * np.eye(n)
    
    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]
        
    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        diff_new = xn[:, None] - x[None, :]
        K_new = np.exp(-diff_new**2 / (2.0 * sigma**2))
        return K_new @ w
        
    return predict


def _fit_poly(X, y, deg):
    """
    Fit polynomial model.
    """
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    
    # Use scaled x for better conditioning
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-10:
        x_std = 1.0
        
    x_scaled = (x - x_mean) / x_std
    V = np.vander(x_scaled, N=deg + 1, increasing=True)
    
    try:
        w = np.linalg.lstsq(V, y, rcond=None)[0]
    except Exception:
        w = np.zeros(deg + 1)
        
    def predict(X_new):
        xn = X_new[:, 0].astype(np.float64)
        xn_scaled = (xn - x_mean) / x_std
        Vn = np.vander(xn_scaled, N=deg + 1, increasing=True)
        return Vn @ w
        
    return predict


def _fit_spline(X, y, s_factor):
    """
    Fit B-spline model.
    """
    from scipy.interpolate import UnivariateSpline
    
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    
    # Sort by x for spline
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    s = s_factor * np.var(y) * n
    
    try:
        sp = UnivariateSpline(x_sorted, y_sorted, s=s, k=4)
        def predict(X_new):
            return sp(X_new[:, 0].astype(np.float64))
        return predict
    except Exception:
        # Fallback to linear
        def predict(X_new):
            return np.full(len(X_new), np.mean(y))
        return predict


def fit(X_train, y_train):
    global _GLOBAL_START
    _GLOBAL_START = time.time()
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n = len(X)
    
    if n < 4:
        c = np.mean(y) if n > 0 else 0.0
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    best_model = None
    best_cv_err = np.inf
    best_name = "none"
    
    # Define candidate models and their hyperparameters
    # 1. RBF with different sigmas
    # Sigma range: from small (local) to large (global)
    # Estimate typical distance
    x = X[:, 0]
    x_range = x.max() - x.min()
    if x_range < 1e-10:
        x_range = 1.0
        
    sigma_candidates = [0.1 * x_range, 0.2 * x_range, 0.3 * x_range, 0.5 * x_range, 0.8 * x_range, 1.2 * x_range]
    
    # 2. Polynomials
    poly_degrees = [2, 3, 4, 5, 6]
    
    # 3. Splines
    spline_factors = [0.01, 0.1, 1.0, 10.0]
    
    # Evaluate candidates
    # RBF
    for sigma in sigma_candidates:
        if _remaining() < 10:
            break
        try:
            fit_func = lambda X_tr, y_tr, s=sigma: _fit_rbf(X_tr, y_tr, s)
            cv_err = _cv_single_model(fit_func, X, y)
            if cv_err < best_cv_err:
                best_cv_err = cv_err
                best_model = _fit_rbf(X, y, sigma)
                best_name = f"RBF(s={sigma:.3f})"
        except Exception:
            pass
            
    # Polynomials
    for deg in poly_degrees:
        if _remaining() < 10:
            break
        try:
            fit_func = lambda X_tr, y_tr, d=deg: _fit_poly(X_tr, y_tr, d)
            cv_err = _cv_single_model(fit_func, X, y)
            if cv_err < best_cv_err:
                best_cv_err = cv_err
                best_model = _fit_poly(X, y, deg)
                best_name = f"Poly(d={deg})"
        except Exception:
            pass
            
    # Splines
    for sf in spline_factors:
        if _remaining() < 10:
            break
        try:
            fit_func = lambda X_tr, y_tr, f=sf: _fit_spline(X_tr, y_tr, f)
            cv_err = _cv_single_model(fit_func, X, y)
            if cv_err < best_cv_err:
                best_cv_err = cv_err
                best_model = _fit_spline(X, y, sf)
                best_name = f"Spline(s={sf})"
        except Exception:
            pass
            
    # If no model was selected (e.g., all failed), use a simple fallback
    if best_model is None:
        # Linear regression fallback
        A = np.column_stack([np.ones(n), X[:, 0]])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            def predict(X_new):
                An = np.column_stack([np.ones(len(X_new)), X_new[:, 0]])
                return An @ w
            best_model = predict
        except Exception:
            c = np.mean(y)
            def predict(X_new):
                return np.full(len(X_new), c)
            best_model = predict

    def safe_predict(X_new):
        try:
            r = best_model(np.asarray(X_new, dtype=np.float64))
            r = np.asarray(r, dtype=np.float64).flatten()
            r = np.nan_to_num(r, nan=0.0, posinf=1e6, neginf=-1e6)
            return r
        except Exception:
            return np.zeros(len(X_new))

    return safe_predict
# EVOLVE-BLOCK-END
