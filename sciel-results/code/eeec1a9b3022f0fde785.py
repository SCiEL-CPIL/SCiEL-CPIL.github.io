"""Combined symbolic regression: Rich Basis + CV Ridge vs Spline Interpolation."""
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
    
    # Pairwise interactions for multi-dim (d is small here)
    if d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
    
    return np.column_stack(cols)

def _clean_features(Phi):
    """Remove non-finite and low-variance columns. Returns (Phi_clean, mask_for_original)."""
    n_feat = Phi.shape[1]
    finite_cols = np.all(np.isfinite(Phi), axis=0)
    # Build mask for original Phi
    mask = finite_cols.copy()
    Phi_finite = Phi[:, finite_cols]
    if Phi_finite.shape[1] == 0:
        return Phi_finite, mask
    col_var = np.var(Phi_finite, axis=0)
    keep_finite = col_var > 1e-12
    # Map back to original indices
    finite_indices = np.where(finite_cols)[0]
    mask[finite_indices[keep_finite]] = True
    return Phi_finite[:, keep_finite], mask

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
    d = X_train.shape[1]
    
    best_model_type = None
    best_model_data = None
    best_train_score = np.inf
    
    # --- Model 1: Rich Basis + Ridge ---
    try:
        Phi = _build_features(X_train, max_p=10)
        Phi_clean, keep_mask = _clean_features(Phi)
        
        if Phi_clean.shape[1] > 0:
            # Cross-validation for lambda selection
            best_w = None
            best_cv_score = np.inf
            best_lambda = None
            
            lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
            
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
            
            if best_lambda is not None:
                # Final fit with best lambda
                best_w = _ridge_solve(Phi_clean, y_train, best_lambda)
                best_w = np.where(np.isfinite(best_w), best_w, 0.0)
                
                # Calculate training score for comparison
                yhat_train = Phi_clean @ best_w
                yhat_train = np.where(np.isfinite(yhat_train), yhat_train, 0.0)
                train_score = _nrmse(y_train, yhat_train)
                
                if train_score < best_train_score:
                    best_train_score = train_score
                    best_model_type = 'ridge'
                    best_model_data = {
                        'w': best_w,
                        'keep_mask': keep_mask,
                        'y_mean': float(np.mean(y_train))
                    }
    except Exception:
        pass
    
    # --- Model 2: Spline (only for 1D inputs) ---
    if d == 1:
        try:
            x = X_train[:, 0]
            y = y_train
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            
            # Try cubic spline interpolation
            try:
                spline = UnivariateSpline(x_sorted, y_sorted, s=0, k=3)
                yhat_train = spline(x)
                if np.all(np.isfinite(yhat_train)):
                    train_score = _nrmse(y, yhat_train)
                    if train_score < best_train_score:
                        best_train_score = train_score
                        best_model_type = 'spline'
                        best_model_data = {
                            'spline': spline
                        }
            except Exception:
                pass
            
            # Try RBF interpolation as alternative
            try:
                # Use all training points as centers
                centers = x_sorted.reshape(-1, 1)
                # Determine width: median distance between consecutive points
                if len(x_sorted) > 1:
                    diffs = np.diff(x_sorted)
                    diffs = diffs[diffs > 1e-10]
                    if len(diffs) > 0:
                        width = float(np.median(diffs))
                    else:
                        width = 0.01
                else:
                    width = 0.01
                width = max(width, 1e-6)
                
                # Build RBF matrix
                dist_sq = (x_sorted[:, None] - x_sorted[None, :]) ** 2
                K = np.exp(-dist_sq / (2.0 * width * width))
                # Add regularization for numerical stability
                K += 1e-10 * np.eye(len(x_sorted))
                
                try:
                    w_rbf = np.linalg.solve(K, y_sorted)
                    # Check training fit
                    yhat_train_rbf = K @ w_rbf
                    if np.all(np.isfinite(yhat_train_rbf)):
                        train_score_rbf = _nrmse(y, yhat_train_rbf)
                        if train_score_rbf < best_train_score:
                            best_train_score = train_score_rbf
                            best_model_type = 'rbf'
                            best_model_data = {
                                'centers': centers,
                                'width': width,
                                'w': w_rbf
                            }
                except np.linalg.LinAlgError:
                    pass
            except Exception:
                pass
            
            # Try PCHIP (monotone cubic) as another option
            try:
                from scipy.interpolate import PchipInterpolator
                pchip = PchipInterpolator(x_sorted, y_sorted)
                yhat_train_p = pchip(x)
                if np.all(np.isfinite(yhat_train_p)):
                    train_score_p = _nrmse(y, yhat_train_p)
                    if train_score_p < best_train_score:
                        best_train_score = train_score_p
                        best_model_type = 'pchip'
                        best_model_data = {
                            'pchip': pchip
                        }
            except Exception:
                pass
        except Exception:
            pass
    
    # --- Fallback: Mean ---
    if best_model_type is None:
        m = float(np.mean(y_train))
        return lambda X: np.full(X.shape[0], m)
    
    y_mean = float(np.mean(y_train))
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        n_test = X.shape[0]
        
        if best_model_type == 'ridge':
            try:
                P = _build_features(X, max_p=10)
                P_clean = P[:, best_model_data['keep_mask']]
                yhat = P_clean @ best_model_data['w']
                yhat = np.where(np.isfinite(yhat), yhat, best_model_data['y_mean'])
                return yhat.astype(np.float64)
            except Exception:
                return np.full(n_test, best_model_data['y_mean'], dtype=np.float64)
        
        elif best_model_type == 'spline':
            try:
                x = X[:, 0]
                yhat = best_model_data['spline'](x)
                yhat = np.where(np.isfinite(yhat), yhat, y_mean)
                return yhat.astype(np.float64)
            except Exception:
                return np.full(n_test, y_mean, dtype=np.float64)
        
        elif best_model_type == 'rbf':
            try:
                x = X[:, 0].reshape(-1, 1)
                centers = best_model_data['centers']
                width = best_model_data['width']
                w = best_model_data['w']
                dist_sq = (x - centers) ** 2
                K_test = np.exp(-dist_sq / (2.0 * width * width))
                yhat = K_test @ w
                yhat = np.where(np.isfinite(yhat), yhat, y_mean)
                return yhat.astype(np.float64)
            except Exception:
                return np.full(n_test, y_mean, dtype=np.float64)
        
        elif best_model_type == 'pchip':
            try:
                x = X[:, 0]
                yhat = best_model_data['pchip'](x)
                yhat = np.where(np.isfinite(yhat), yhat, y_mean)
                return yhat.astype(np.float64)
            except Exception:
                return np.full(n_test, y_mean, dtype=np.float64)
        
        else:
            return np.full(n_test, y_mean, dtype=np.float64)
    
    return predict

# EVOLVE-BLOCK-END
