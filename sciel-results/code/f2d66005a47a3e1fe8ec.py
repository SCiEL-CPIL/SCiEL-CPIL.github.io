"""Parametric Global Basis Regression for 1-D noiseless equations."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _as_1d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _fit_fourier(x, y, max_harm):
    """Fit y using Fourier basis up to max_harm harmonics."""
    x_min, x_max = np.min(x), np.max(x)
    span = x_max - x_min
    if span < 1e-9:
        return lambda xv: np.full_like(xv, np.mean(y))
    
    # Base frequency: 2*pi/span covers 1 cycle over the domain
    # We assume the function is periodic with period span or fits within it.
    # To be safe for non-periodic, we might need more harmonics or different basis.
    # But for bounded 1D, Fourier is a strong competitor.
    
    # We construct the design matrix A
    # Columns: 1, sin(f1 x), cos(f1 x), sin(f2 x), cos(f2 x), ...
    # Note: Using x directly in sin/cos.
    
    A = np.ones((len(x), 1))
    for k in range(1, max_harm + 1):
        f = k * (2 * np.pi / span)
        A = np.column_stack([A, np.sin(f * x), np.cos(f * x)])
    
    # Ridge regression for stability
    lam = 1e-6
    try:
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, y, rcond=None)[0]
    
    def predict(Xv):
        xv = Xv[:, 0]
        Av = np.ones((len(xv), 1))
        for k in range(1, max_harm + 1):
            f = k * (2 * np.pi / span)
            Av = np.column_stack([Av, np.sin(f * xv), np.cos(f * xv)])
        return Av @ w
    return predict

def _fit_chebyshev(x, y, max_deg):
    """Fit y using Chebyshev polynomial basis."""
    x_min, x_max = np.min(x), np.max(x)
    x_center = (x_min + x_max) / 2.0
    x_scale = (x_max - x_min) / 2.0
    if x_scale < 1e-9:
        return lambda xv: np.full_like(xv, np.mean(y))
    
    # Map to [-1, 1]
    x_norm = (x - x_center) / x_scale
    
    # Chebyshev coefficients
    # Use numpy.polynomial.chebyshev
    try:
        coeffs = np.polynomial.chebyshev.chebfit(x_norm, y, max_deg)
    except Exception:
        # Fallback to poly if chebfit fails
        coeffs = np.zeros(max_deg + 1)
        
    def predict(Xv):
        xv = Xv[:, 0]
        xv_norm = (xv - x_center) / x_scale
        return np.polynomial.chebyshev.chebval(xv_norm, coeffs)
    return predict

def _fit_poly(x, y, max_deg):
    """Fit y using standard polynomial basis."""
    try:
        # Use vandermonde for stability? np.polyfit is okay for low deg.
        # Let's use ridge regression on Vandermonde matrix for better conditioning.
        F = np.vander(x, N=max_deg + 1, increasing=True)
        lam = 1e-6
        w = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ y)
        
        def predict(Xv):
            xv = Xv[:, 0]
            Fv = np.vander(xv, N=max_deg + 1, increasing=True)
            return Fv @ w
        return predict
    except Exception:
        return lambda Xv: np.full_like(Xv[:, 0], np.mean(y))

def _fit_rbf(x, y, gamma, lam):
    """Fit y using Gaussian RBF."""
    K = np.exp(-gamma * (x[:, None] - x[None, :])**2)
    try:
        w = np.linalg.solve(K + lam * np.eye(len(x)), y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K + lam * np.eye(len(x)), y, rcond=None)[0]
        
    def predict(Xv):
        xv = Xv[:, 0]
        K_test = np.exp(-gamma * (xv[:, None] - x[None, :])**2)
        return K_test @ w
    return predict

def fit(X_train, y_train):
    X = _as_1d(X_train)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    n, d = X.shape
    
    if d != 1:
        # Fallback for multi-dim (though task says 1D inputs)
        # Just use linear regression
        X_ext = np.column_stack([np.ones(n), X])
        w = np.linalg.lstsq(X_ext, y, rcond=None)[0]
        def predict(Xv):
            Xv = _as_1d(Xv)
            Xv_ext = np.column_stack([np.ones(len(Xv)), Xv])
            return Xv_ext @ w
        return predict

    x = X[:, 0]
    
    # Validation split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(20, int(0.2 * n))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    
    def val_mse(pred_fn):
        try:
            p = pred_fn(np.column_stack([x_val]))
            p = np.asarray(p, dtype=float).reshape(-1)
            if not np.all(np.isfinite(p)):
                return np.inf
            return np.mean((p - y_val)**2)
        except Exception:
            return np.inf

    candidates = []
    
    # 1. Fourier fits
    for h in [5, 10, 20, 50, 100]:
        try:
            p_fn = _fit_fourier(x_tr, y_tr, h)
            mse = val_mse(p_fn)
            if np.isfinite(mse):
                candidates.append((mse, p_fn))
        except Exception:
            pass
            
    # 2. Chebyshev fits
    for deg in [3, 5, 10, 20, 40]:
        try:
            p_fn = _fit_chebyshev(x_tr, y_tr, deg)
            mse = val_mse(p_fn)
            if np.isfinite(mse):
                candidates.append((mse, p_fn))
        except Exception:
            pass

    # 3. Polynomial fits
    for deg in [2, 4, 6, 10, 20]:
        try:
            p_fn = _fit_poly(x_tr, y_tr, deg)
            mse = val_mse(p_fn)
            if np.isfinite(mse):
                candidates.append((mse, p_fn))
        except Exception:
            pass

    # 4. RBF fits
    for gamma in [0.1, 0.5, 1.0, 5.0, 10.0]:
        for lam in [1e-6, 1e-4, 1e-2]:
            try:
                p_fn = _fit_rbf(x_tr, y_tr, gamma, lam)
                mse = val_mse(p_fn)
                if np.isfinite(mse):
                    candidates.append((mse, p_fn))
            except Exception:
                pass

    if not candidates:
        # Fallback: Mean
        c = np.mean(y)
        return lambda Xv: np.full(_as_1d(Xv).shape[0], c)

    # Select best
    candidates.sort(key=lambda t: t[0])
    best_mse, best_pred = candidates[0]
    
    # If the best validation error is still high, it might be a good idea to 
    # re-fit the best *type* on the full data.
    # However, for parametric models, fitting on full data is standard.
    # The selection was done on validation. Now we re-fit the winner's structure on all data.
    
    # Identify which model type won to re-fit on full data
    # This is tricky with the closure structure. 
    # Alternative: Just use the model fitted on train. 
    # For 200 points, train vs full data difference is small.
    # But re-fitting on full data is slightly better for test.
    
    # Let's just use the best model found. It's robust enough.
    
    def predict(X):
        Xv = _as_1d(X)
        try:
            out = best_pred(Xv)
            out = np.asarray(out, dtype=float).reshape(-1)
            if out.shape[0] != Xv.shape[0]:
                out = np.resize(out, Xv.shape[0])
            out = np.nan_to_num(out, nan=np.mean(y), posinf=1e6, neginf=-1e6)
            return out
        except Exception:
            return np.full(Xv.shape[0], np.mean(y))
            
    return predict
# EVOLVE-BLOCK-END
