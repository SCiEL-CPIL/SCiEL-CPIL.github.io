"""Combined symbolic regression: rich basis + cross-validated ridge + spline basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import UnivariateSpline

def _build_features(X, max_p=10):
    """Build a rich feature matrix from input X (n, d)."""
    n = X.shape[0]
    d = X.shape[1]
    cols = [np.ones(n, dtype=np.float64)]
    
    for j in range(d):
        x = X[:, j].astype(np.float64)
        ax = np.abs(x)
        
        # Polynomials x^1 .. x^max_p
        powv = x.copy()
        for p in range(1, max_p + 1):
            cols.append(powv)
            if p < max_p:
                powv = powv * x
        
        # Abs and sqrt of abs
        cols.append(ax)
        sqrt_ax = np.sqrt(ax)
        cols.append(sqrt_ax)
        cols.append(sqrt_ax * x)  # x * |x|^0.5
        cols.append(ax ** 0.25)
        
        # Reciprocal-ish (guarded)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.where(np.abs(x) > 1e-6, 1.0 / x, 0.0)
        cols.append(inv)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv2 = np.where(np.abs(x) > 1e-6, 1.0 / (x * x), 0.0)
        cols.append(inv2)
        
        # Trigonometric
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.sin(x * x))
        cols.append(np.cos(x * x))
        cols.append(np.sin(np.sqrt(ax)))
        
        # Exponential (clipped to avoid overflow)
        xc = np.clip(x, -50, 50)
        cols.append(np.exp(xc))
        cols.append(np.exp(-xc))
        cols.append(np.exp(-x * x))  # Gaussian-like
        cols.append(np.exp(x * x * 0.5))
        
        # Logarithms (guarded)
        cols.append(np.where(x > 1e-9, np.log(x), 0.0))
        cols.append(np.log1p(np.abs(x)))
        cols.append(np.where(ax > 1e-9, np.log(ax), 0.0))
        
        # Power laws (non-integer)
        cols.append(x * np.sqrt(ax))
        cols.append(x * ax * sqrt_ax)  # x |x|^1.5
        cols.append(np.sign(x) * (ax ** 0.75))
        cols.append(np.sign(x) * (ax ** 1.25))
        cols.append(np.sign(x) * (ax ** 0.5))
        cols.append(np.sign(x) * (ax ** 1.5))
        cols.append(np.sign(x) * (ax ** 2.5))
        
        # sinh, cosh, sech
        sc = np.clip(xc, -20, 20)
        cols.append(np.sinh(sc))
        cols.append(np.cosh(sc))
        with np.errstate(over='ignore'):
            sech = 1.0 / np.cosh(sc)
            sech[~np.isfinite(sech)] = 0.0
        cols.append(sech)
        
        # Spline basis functions (B-spline-like using cubic B-splines via scipy)
        try:
            # Create knot points
            xmin, xmax = x.min(), x.max()
            if xmax - xmin > 1e-10:
                n_knots = 8
                knots = np.linspace(xmin, xmax, n_knots + 4)
                # Use scipy's BSpline basis
                from scipy.interpolate import BSpline
                # Build cubic B-spline basis
                for k in range(n_knots):
                    try:
                        b = BSpline.basis_element(k)
                        # Evaluate at shifted/scaled x
                        xs = (x - knots[0]) / (knots[-1] - knots[0]) * (n_knots - 1)
                        bvals = b(xs)
                        bvals = np.where(np.isfinite(bvals), bvals, 0.0)
                        cols.append(bvals)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Pairwise interactions for multi-dim (d is small here)
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
    
    return np.column_stack(cols)

def _clean_features(Phi, y):
    """Remove non-finite and low-variance columns."""
    finite_cols = np.all(np.isfinite(Phi), axis=0)
    Phi_clean = Phi[:, finite_cols]
    if Phi_clean.shape[1] == 0:
        return Phi, np.zeros(Phi.shape[1], dtype=bool)
    col_var = np.var(Phi_clean, axis=0)
    keep = col_var > 1e-12
    return Phi_clean[:, keep], keep

def _ridge_solve(Phi, y, lam):
    """Solve ridge regression for given lambda."""
    nfeat = Phi.shape[1]
    if lam == 0.0:
        w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    else:
        A = Phi.T @ Phi + lam * np.eye(nfeat)
        b = Phi.T @ y
        try:
            L = np.linalg.cholesky(A)
            w = np.linalg.solve(L.T, np.linalg.solve(L, b))
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w

def _nrmse(y, yhat):
    """Compute normalized MSE."""
    resid = y - yhat
    mse = np.mean(resid * resid)
    vy = np.var(y)
    if vy <= 1e-12:
        return mse
    return mse / vy

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 290.0  # safety margin under 300s
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n = X_train.shape[0]
    
    # Build features once for training
    try:
        Phi = _build_features(X_train, max_p=10)
    except Exception:
        # Fallback: simple degree-3 poly
        d = X_train.shape[1]
        cols = [np.ones(n)]
        for j in range(d):
            x = X_train[:, j]
            cols.append(x); cols.append(x*x); cols.append(x*x*x)
        Phi = np.column_stack(cols)
    
    # Clean features
    Phi_clean, keep_mask = _clean_features(Phi, y_train)
    
    if Phi_clean.shape[1] == 0:
        m = float(np.mean(y_train))
        return lambda X: np.full(X.shape[0], m)
    
    # Cross-validation for lambda selection
    best_w = None
    best_cv_score = np.inf
    best_lambda = None
    
    lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    # Simple 5-fold CV
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    for lam in lambdas:
        if time.time() > DEADLINE - 10:
            break
        cv_scores = []
        try:
            for fold in range(n_folds):
                val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
                train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
                
                Phi_tr = Phi_clean[train_idx]
                y_tr = y_train[train_idx]
                Phi_val = Phi_clean[val_idx]
                y_val = y_train[val_idx]
                
                w = _ridge_solve(Phi_tr, y_tr, lam)
                yhat_val = Phi_val @ w
                if not np.all(np.isfinite(yhat_val)):
                    cv_scores.append(np.inf)
                    continue
                cv_scores.append(_nrmse(y_val, yhat_val))
            
            mean_cv = np.mean(cv_scores)
            if mean_cv < best_cv_score:
                best_cv_score = mean_cv
                best_lambda = lam
        except Exception:
            continue
    
    # If CV didn't find a good lambda, fall back to training NRMSE
    if best_lambda is None:
        for lam in lambdas:
            if time.time() > DEADLINE - 5:
                break
            try:
                w = _ridge_solve(Phi_clean, y_train, lam)
                yhat = Phi_clean @ w
                if not np.all(np.isfinite(yhat)):
                    continue
                nm = _nrmse(y_train, yhat)
                if nm < best_cv_score:
                    best_cv_score = nm
                    best_lambda = lam
            except Exception:
                continue
    
    # Final fit with best lambda
    if best_lambda is None:
        best_lambda = 1e-6
    
    try:
        best_w = _ridge_solve(Phi_clean, y_train, best_lambda)
    except Exception:
        m = float(np.mean(y_train))
        return lambda X: np.full(X.shape[0], m)
    
    # Ensure finite weights
    best_w = np.where(np.isfinite(best_w), best_w, 0.0)
    
    y_mean = float(np.mean(y_train))
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        n_test = X.shape[0]
        try:
            P = _build_features(X, max_p=10)
            P, _ = _clean_features(P, y_train)
            yhat = P @ best_w
            yhat = np.where(np.isfinite(yhat), yhat, y_mean)
            return yhat.astype(np.float64)
        except Exception:
            return np.full(n_test, y_mean, dtype=np.float64)
    
    return predict

# EVOLVE-BLOCK-END
