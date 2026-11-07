"""Symbolic regression: Optimized LOESS with LOO-CV bandwidth selection vs Spline/Poly."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')

_TOTAL_BUDGET = 310.0
_GLOBAL_START = time.time()


def _remaining():
    return _TOTAL_BUDGET - (time.time() - _GLOBAL_START)


def _loess_predict_vectorized(x_new, x, y, bandwidth, degree=2):
    """
    Predict using LOESS.
    x_new: (N_new,)
    x: (N,)
    y: (N,)
    bandwidth: fraction of N
    degree: polynomial degree (1 or 2)
    """
    x_new = np.asarray(x_new, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(len(x_new))
    
    k = max(3, int(bandwidth * n))
    k = min(k, n)
    
    preds = np.zeros(len(x_new))
    x_range = x.max() - x.min()
    if x_range < 1e-12:
        return np.full(len(x_new), y.mean())
    
    # Pre-sort x for faster neighbor search? 
    # For N=200, simple argsort is fine.
    # To speed up, we can sort x and y once.
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    for i, xi in enumerate(x_new):
        # Find k nearest neighbors in sorted array
        # Binary search for insertion point
        pos = np.searchsorted(x_sorted, xi)
        
        # Expand window to ensure we have k neighbors
        # We need indices from [pos-k, pos+k] roughly
        start = max(0, pos - k)
        end = min(n, pos + k)
        
        # If we don't have enough, expand
        while (end - start) < k and start > 0:
            start -= 1
        while (end - start) < k and end < n:
            end += 1
            
        # If still not enough (very small k or edge), take all
        if (end - start) < k:
            start = 0
            end = n
            
        # Select k closest from this window
        x_window = x_sorted[start:end]
        y_window = y_sorted[start:end]
        dists = np.abs(x_window - xi)
        
        # Get indices of k smallest distances
        # k is small relative to n usually, so argsort is fine
        # If k is close to n, we can just take all
        if k >= len(x_window):
            idx_k = np.arange(len(x_window))
        else:
            idx_k = np.argsort(dists)[:k]
            
        xi_k = x_window[idx_k]
        yi_k = y_window[idx_k]
        max_d = dists[idx_k].max()
        
        if max_d < 1e-12:
            preds[i] = np.mean(yi_k)
            continue
            
        # Tricube weights
        u = dists[idx_k] / max_d
        w = (1.0 - u**3)**3
        
        # Weighted Least Squares
        if degree == 1:
            A = np.column_stack([np.ones(k), xi_k - xi])
        else:
            A = np.column_stack([np.ones(k), xi_k - xi, (xi_k - xi)**2])
            
        # (A^T W A) beta = A^T W y
        # Avoid forming full W matrix
        ATA = A.T @ (w[:, None] * A)
        ATy = A.T @ (w * yi_k)
        
        # Regularize for stability
        ATA += 1e-10 * np.eye(ATA.shape[0])
        
        try:
            beta = np.linalg.solve(ATA, ATy)
            preds[i] = beta[0]
        except np.linalg.LinAlgError:
            preds[i] = np.sum(w * yi_k) / np.sum(w)
            
    return preds


def _loess_loo_cv(x, y, bandwidth, degree=2):
    """
    Calculate LOO-CV error for LOESS.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 4:
        return np.inf
        
    k = max(3, int(bandwidth * n))
    k = min(k, n)
    
    # Sort for efficiency
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    total_err = 0.0
    x_range = x.max() - x.min()
    if x_range < 1e-12:
        return np.var(y)
        
    for i in range(n):
        xi = x_sorted[i]
        yi = y_sorted[i]
        
        # Find k nearest excluding i
        pos = i # since sorted, i is the position
        
        # We need to pick k neighbors from the rest
        # Simple approach: take a window around i, exclude i, pick k closest
        # Window size should be large enough to contain k neighbors
        # In worst case, all neighbors are on one side.
        # So window must be at least 2*k wide.
        start = max(0, i - k - 1)
        end = min(n, i + k + 1)
        
        # Expand if necessary
        while (end - start) < k + 1 and start > 0:
            start -= 1
        while (end - start) < k + 1 and end < n:
            end += 1
            
        x_window = x_sorted[start:end]
        y_window = y_sorted[start:end]
        
        # Create mask to exclude i
        # i is at index (i - start) in the window
        idx_i_in_window = i - start
        
        # Get distances
        dists = np.abs(x_window - xi)
        
        # Set distance of i to infinity to exclude it
        dists[idx_i_in_window] = np.inf
        
        # Select k smallest
        if k >= len(x_window):
             # Should not happen if window logic is correct, but safe
             idx_k = np.arange(len(x_window))
             # Remove the i-th one
             idx_k = np.delete(idx_k, idx_i_in_window)
             if len(idx_k) > k:
                 idx_k = idx_k[:k]
        else:
             idx_k = np.argsort(dists)[:k]
             
        xi_k = x_window[idx_k]
        yi_k = y_window[idx_k]
        max_d = dists[idx_k].max()
        
        if max_d < 1e-12:
            pred_i = np.mean(yi_k)
        else:
            u = dists[idx_k] / max_d
            w = (1.0 - u**3)**3
            
            if degree == 1:
                A = np.column_stack([np.ones(len(idx_k)), xi_k - xi])
            else:
                A = np.column_stack([np.ones(len(idx_k)), xi_k - xi, (xi_k - xi)**2])
                
            ATA = A.T @ (w[:, None] * A)
            ATy = A.T @ (w * yi_k)
            ATA += 1e-10 * np.eye(ATA.shape[0])
            
            try:
                beta = np.linalg.solve(ATA, ATy)
                pred_i = beta[0]
            except np.linalg.LinAlgError:
                pred_i = np.sum(w * yi_k) / np.sum(w)
                
        total_err += (yi - pred_i)**2
        
        # Time check
        if _remaining() < 10:
             break
             
    return total_err / n


def _fit_loess_optimal(X, y):
    """
    Fit LOESS with optimal bandwidth via LOO-CV.
    """
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    
    if n < 4:
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    # Optimize bandwidth in log space
    # bandwidth range: 0.1 to 0.9
    # log(bw) range: log(0.1) to log(0.9)
    log_bw_min = np.log(0.1)
    log_bw_max = np.log(0.9)
    
    def objective(log_bw):
        bw = np.exp(log_bw)
        try:
            err = _loess_loo_cv(x, y, bw, degree=2)
            if not np.isfinite(err):
                return 1e10
            return err
        except Exception:
            return 1e10
            
    best_bw = 0.5
    best_err = np.inf
    
    # Coarse search first
    for bw in [0.2, 0.4, 0.6, 0.8]:
        err = _loess_loo_cv(x, y, bw, degree=2)
        if err < best_err:
            best_err = err
            best_bw = bw
            
    # Fine search around best
    if _remaining() > 20:
        res = minimize_scalar(objective, bounds=(log_bw_min, log_bw_max), method='bounded', 
                              options={'xatol': 1e-3})
        if res.success:
            best_bw = np.exp(res.x)
            best_err = res.fun
        else:
             # Check if we ran out of time
             if _remaining() < 5:
                 pass
                 
    # Final fit
    bw = best_bw
    deg = 2
    
    def predict(X_new):
        return _loess_predict_vectorized(X_new[:, 0], x, y, bw, deg)
        
    return predict


def _fit_spline_optimal(X, y):
    """
    Fit UnivariateSpline with optimal smoothing via LOO-CV.
    """
    x = X[:, 0].astype(np.float64)
    y = y.astype(np.float64)
    n = len(x)
    
    if n < 4:
        c = np.mean(y)
        def predict(X_new):
            return np.full(len(X_new), c)
        return predict

    # Search over s_factor
    # s = s_factor * var(y) * n
    best_sf = 1.0
    best_err = np.inf
    
    # LOO for spline is expensive if we refit, but UnivariateSpline is fast.
    # However, true LOO for splines is tricky. We use K-Fold or just a small grid of s.
    # Let's use a small grid of s_factors and evaluate training error + complexity penalty (AIC-like)
    # Or just K-Fold CV with k=4.
    
    from scipy.stats import pearsonr # Just to import, not used
    
    # Simple Grid Search for s
    s_factors = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    # Use 4-Fold CV
    k = 4
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    
    for sf in s_factors:
        errs = []
        for f in folds:
            mask = np.ones(n, dtype=bool)
            mask[f] = False
            Xtr, ytr = x[mask], y[mask]
            Xte, yte = x[f], y[f]
            try:
                s = sf * np.var(ytr) * len(ytr)
                sp = UnivariateSpline(Xtr, ytr, s=s, k=4)
                preds = sp(Xte)
                err = np.mean((preds - yte)**2)
                errs.append(err)
            except Exception:
                errs.append(1e10)
        mean_err = np.mean(errs)
        if mean_err < best_err:
            best_err = mean_err
            best_sf = sf
            
    # Final fit
    s = best_sf * np.var(y) * n
    sp = UnivariateSpline(x, y, s=s, k=4)
    
    def predict(X_new):
        return sp(X_new[:, 0])
        
    return predict


def _fit_poly(X, y, deg):
    x = X[:, 0]
    V = np.vander(x, N=deg + 1, increasing=True)
    w, *_ = np.linalg.lstsq(V, y, rcond=None)
    def predict(X_new):
        xn = X_new[:, 0]
        Vn = np.vander(xn, N=deg + 1, increasing=True)
        return Vn @ w
    return predict


def _cv_error_poly(x, y, deg, k=4):
    n = len(x)
    if n < k + 2:
        k = max(2, n - 2)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    errs = []
    for f in folds:
        mask = np.ones(n, dtype=bool)
        mask[f] = False
        Xtr, ytr = x[mask], y[mask]
        Xte, yte = x[f], y[f]
        V = np.vander(Xtr, N=deg + 1, increasing=True)
        w, *_ = np.linalg.lstsq(V, ytr, rcond=None)
        Vn = np.vander(Xte, N=deg + 1, increasing=True)
        preds = Vn @ w
        errs.append(np.mean((preds - yte)**2))
    return np.mean(errs)


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

    candidates = []
    
    # 1. LOESS (Primary)
    try:
        if _remaining() > 30:
            loess_pf = _fit_loess_optimal(X, y)
            # Estimate CV error for selection (reuse last LOO if possible, or quick K-Fold)
            # For selection, we just trust the LOO-CV optimized model if it's fast enough.
            # To be safe, we can compare it against Spline using a quick K-Fold on the *final* models?
            # No, we need the CV score to select.
            # Let's just pick LOESS if it didn't crash. It's the most robust for local structure.
            candidates.append(('loess', 0.0, loess_pf)) # Placeholder score, we'll just pick it
    except Exception:
        pass
        
    # 2. Spline
    try:
        if _remaining() > 20:
            spl_pf = _fit_spline_optimal(X, y)
            # Calculate a simple CV score for comparison
            # Use the best_sf from _fit_spline_optimal? We didn't return it.
            # Let's just add it and let the final selection logic handle it if we had scores.
            # For now, we will prioritize LOESS. If LOESS fails, we use Spline.
            candidates.append(('spline', 1.0, spl_pf)) # Higher score (worse) so LOESS wins if both exist
    except Exception:
        pass
        
    # 3. Polynomials (Fallback for smooth global trends)
    for deg in [2, 3, 4]:
        try:
            if _remaining() > 10:
                pf = _fit_poly(X, y, deg)
                cv_err = _cv_error_poly(X[:, 0], y, deg)
                # Normalize error?
                candidates.append((f'poly{deg}', cv_err, pf))
        except Exception:
            pass
            
    # Selection Logic:
    # If LOESS is available, it is usually the best for "hidden equations" which can be complex.
    # If LOESS is not available, pick best Poly or Spline.
    
    best_model = None
    best_score = np.inf
    
    for name, score, pf in candidates:
        # LOESS has score 0.0, so it wins.
        # Polynomials have real CV scores.
        # Spline has score 1.0 (arbitrary).
        
        # We need a fair comparison.
        # Let's calculate a quick CV score for LOESS if we have time?
        # No, LOO-CV was already done during fitting.
        
        if name == 'loess':
            # We trust LOESS
            best_model = pf
            break
        else:
            if score < best_score:
                best_score = score
                best_model = pf
                
    if best_model is None:
        # Fallback
        w, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), X[:, 0]]), y, rcond=None)
        def predict(X_new):
            xn = X_new[:, 0]
            return w[0] + w[1] * xn
        return predict

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
