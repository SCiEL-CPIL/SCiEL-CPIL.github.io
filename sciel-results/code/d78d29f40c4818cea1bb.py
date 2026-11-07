import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import PchipInterpolator, make_interp_spline, UnivariateSpline

_TOTAL_DEADLINE = time.time() + 320.0  # Leave 10s margin for 330s limit

def _remaining():
    return _TOTAL_DEADLINE - time.time()

def _safe_predict(make_fn):
    def predict(X):
        try:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = make_fn(X)
            y = np.asarray(y, dtype=np.float64)
            y = np.nan_to_num(y, nan=0.0, posinf=1e12, neginf=-1e12)
            return y
        except Exception:
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros(X.shape[0])
    return predict

def _fit_pchip(X, y):
    """Fit PCHIP Interpolator."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        # Check for duplicate x values, which can break splines
        # If duplicates exist, average the y values
        unique_xs, indices, counts = np.unique(xs, return_index=True, return_counts=True)
        if len(unique_xs) < len(xs):
            ys_unique = np.array([ys[indices[i]:indices[i]+counts[i]].mean() for i in range(len(unique_xs))])
            # Actually, simpler approach: just use the first occurrence or average properly
            # PchipInterpolator handles duplicates poorly. Let's aggregate.
            xs = unique_xs
            ys = np.zeros_like(xs)
            for i in range(len(xs)):
                mask = (x == xs[i])
                ys[i] = y[mask].mean()
            # Re-sort just in case
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            
        pchip = PchipInterpolator(xs, ys)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = pchip(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None

def _fit_cubic_spline(X, y):
    """Fit Cubic Spline Interpolator."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        unique_xs, indices, counts = np.unique(xs, return_index=True, return_counts=True)
        if len(unique_xs) < len(xs):
            xs = unique_xs
            ys = np.zeros_like(xs)
            for i in range(len(xs)):
                mask = (x == xs[i])
                ys[i] = y[mask].mean()
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            
        if len(xs) < 4:
            return None
            
        spline = make_interp_spline(xs, ys, k=3)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = spline(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None

def _fit_smooth_spline(X, y, s_factor=1e-10):
    """Fit UnivariateSpline with very low smoothing (near interpolation)."""
    try:
        x = X[:, 0]
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        
        unique_xs, indices, counts = np.unique(xs, return_index=True, return_counts=True)
        if len(unique_xs) < len(xs):
            xs = unique_xs
            ys = np.zeros_like(xs)
            for i in range(len(xs)):
                mask = (x == xs[i])
                ys[i] = y[mask].mean()
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            
        n = len(xs)
        # s is the sum of squared residuals. For noiseless, we want it small.
        # UnivariateSpline might fail if s is too small relative to variance.
        # Using a very small s forces it to interpolate closely.
        s = s_factor * np.sum((ys - np.mean(ys))**2)
        if s < 1e-12:
            s = 1e-12
            
        try:
            sp = UnivariateSpline(xs, ys, s=s, k=3)
        except:
            # Fallback to higher smoothing
            sp = UnivariateSpline(xs, ys, s=1.0, k=3)
            
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            yp = sp(xp)
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None

def _fit_poly(X, y, degree):
    """Direct polynomial fit. Good for simple functions."""
    try:
        x = X[:, 0]
        # Use least squares to avoid numerical issues with polyfit for higher degrees
        V = np.vander(x, N=degree+1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(V, y, rcond=None)
        
        def predict(Xp):
            Xp = np.asarray(Xp, dtype=np.float64)
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1)
            xp = Xp[:, 0]
            Vp = np.vander(xp, N=degree+1, increasing=True)
            yp = Vp @ coeffs
            return np.nan_to_num(yp, nan=0.0, posinf=1e12, neginf=-1e12)
        return predict
    except Exception:
        return None

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    n, d = X_train.shape

    # Deterministic validation split
    # Use a fixed fraction for stability
    val_frac = 0.15
    val_size = int(n * val_frac)
    if val_size < 10:
        val_size = 10
    if val_size > n - 10:
        val_size = n - 10
        
    # Use a fixed seed for permutation to ensure reproducibility
    rng = np.random.RandomState(123)
    indices = rng.permutation(n)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    
    var_y_val = np.var(y_val)
    if var_y_val < 1e-12:
        var_y_val = 1.0

    best_predict = None
    best_val_nmse = np.inf

    # Baseline: Mean
    const = np.mean(y_tr)
    const_pred = np.full(X_val.shape[0], const)
    const_val_nmse = np.mean((const_pred - y_val) ** 2) / var_y_val
    best_val_nmse = const_val_nmse
    best_predict = _safe_predict(lambda Xp: np.full(np.asarray(Xp).shape[0], const))

    methods = []
    
    # 1. PCHIP (Top candidate for noiseless smooth 1D)
    methods.append(("pchip", lambda X, y: _fit_pchip(X, y)))
    
    # 2. Cubic Spline
    methods.append(("cubic_spline", lambda X, y: _fit_cubic_spline(X, y)))
    
    # 3. Smooth Splines (varying smoothing)
    for sf in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        methods.append(("smooth_spline_%s" % sf, lambda X, y, sf=sf: _fit_smooth_spline(X, y, sf)))
        
    # 4. Polynomials (low degree to avoid overfitting, high degree to catch complexity)
    for deg in [2, 3, 4, 5, 6, 8, 10, 12]:
        methods.append(("poly_%d" % deg, lambda X, y, d=deg: _fit_poly(X, y, d)))

    # Evaluate methods
    for name, method in methods:
        if _remaining() < 5.0:
            break
            
        try:
            predict = method(X_tr, y_tr)
            if predict is None:
                continue
                
            val_pred = predict(X_val)
            val_pred = np.asarray(val_pred, dtype=np.float64)
            
            if not np.all(np.isfinite(val_pred)):
                continue
                
            val_nmse = np.mean((val_pred - y_val) ** 2) / var_y_val
            
            if not np.isfinite(val_nmse):
                continue
                
            if val_nmse < best_val_nmse:
                best_val_nmse = val_nmse
                # Refit on full data
                full_predict = method(X_train, y_train)
                if full_predict is not None:
                    # Sanity check
                    try:
                        full_pred = full_predict(X_train)
                        if np.all(np.isfinite(full_pred)):
                            best_predict = full_predict
                    except:
                        best_predict = predict
                else:
                    best_predict = predict
                    
        except Exception:
            continue

    # Final check
    if best_predict is None:
        best_predict = _safe_predict(lambda Xp: np.full(np.asarray(Xp).shape[0], const))

    return best_predict
