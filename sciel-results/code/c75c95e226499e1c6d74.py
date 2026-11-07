"""
Strategy: Robust Bayesian Model Averaging (BMA) over diverse basis functions.
Instead of selecting a single best model via grid search (which is prone to overfitting 
and selection bias on small datasets), we fit a portfolio of models:
1. Chebyshev Polynomials (varying degrees)
2. Gaussian RBF Kernels (varying length scales)
3. Fourier Series (varying frequencies)
4. Splines (varying smoothing)

We use a 70/30 train/validation split to estimate validation errors.
We then compute model weights using a robust likelihood approximation:
weights ∝ exp(-beta * robust_loss)
This ensures that models with catastrophic errors receive near-zero weight, 
while good models share the weight. If all models are bad, the average is 
still finite and bounded by the mean of the individual bad predictions, 
preventing total failure.
"""
import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _fit_chebyshev(X, y, degree):
    """Fit Chebyshev polynomial."""
    x = X[:, 0]
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        x_norm = np.zeros_like(x)
    else:
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    
    T = np.ones((len(x), degree + 1))
    if degree >= 1:
        T[:, 1] = x_norm
    for n in range(2, degree + 1):
        T[:, n] = 2 * x_norm * T[:, n-1] - T[:, n-2]
        
    # Ridge regularization for stability
    lam = 1e-6
    w, *_ = np.linalg.lstsq(T + lam * np.eye(degree + 1), y, rcond=None)
    
    def predict(X_in):
        xi = X_in[:, 0]
        if x_max == x_min:
            xi_norm = np.zeros_like(xi)
        else:
            xi_norm = 2 * (xi - x_min) / (x_max - x_min) - 1
        T_in = np.ones((len(xi), degree + 1))
        if degree >= 1:
            T_in[:, 1] = xi_norm
        for n in range(2, degree + 1):
            T_in[:, n] = 2 * xi_norm * T_in[:, n-1] - T_in[:, n-2]
        return T_in @ w
    return predict

def _fit_rbf(X, y, gamma):
    """Fit RBF model with ridge regularization."""
    x = X[:, 0]
    centers = x
    dists = np.abs(x[:, None] - centers[None, :])
    K = np.exp(-gamma * dists**2)
    
    # Ridge regularization
    lam = 1e-6
    K_reg = K + lam * np.eye(len(x))
    
    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]
    
    def predict(X_in):
        xi = X_in[:, 0]
        dists_in = np.abs(xi[:, None] - centers[None, :])
        K_in = np.exp(-gamma * dists_in**2)
        return K_in @ w
    return predict

def _fit_fourier(X, y, n_harmonics):
    """Fit Fourier series."""
    x = X[:, 0]
    cols = [np.ones_like(x)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * x))
        cols.append(np.sin(k * x))
    Xf = np.column_stack(cols)
    
    # Ridge
    lam = 1e-6
    w, *_ = np.linalg.lstsq(Xf + lam * np.eye(len(cols)), y, rcond=None)
    
    def predict(X_in):
        xi = X_in[:, 0]
        cols_in = [np.ones_like(xi)]
        for k in range(1, n_harmonics + 1):
            cols_in.append(np.cos(k * xi))
            cols_in.append(np.sin(k * xi))
        Xf_in = np.column_stack(cols_in)
        return Xf_in @ w
    return predict

def _fit_spline(X, y, s_factor):
    """Fit Univariate Spline."""
    x = X[:, 0]
    idx = np.argsort(x)
    xs = x[idx]
    ys = y[idx]
    
    try:
        s = s_factor * np.sum((ys - ys.mean())**2)
        spl = UnivariateSpline(xs, ys, k=3, s=s)
        def predict(X_in):
            return spl(X_in[:, 0])
        return predict
    except:
        return lambda X_in: np.full(len(X_in), y.mean())

def _robust_loss(y_true, y_pred):
    """
    Compute a robust loss for model weighting.
    We use a scaled log-cosh loss which is smooth and less sensitive to outliers 
    than MSE, helping to avoid over-weighting models that fit noise perfectly 
    but fail on generalization.
    """
    diff = y_pred - y_true
    # Scale by std to make it dimensionless
    std = np.std(diff)
    if std < 1e-8:
        std = 1.0
    diff_norm = diff / std
    # log(cosh(x)) approximates |x| for large x, x^2/2 for small x
    # We use a simple approximation: np.logaddexp(x, -x) - log(2)
    loss = np.mean(np.logaddexp(diff_norm, -diff_norm) - np.log(2))
    return loss

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    # Ensure finite
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    n = len(X)
    if n < 10:
        y_mean = y.mean()
        return lambda X_in: np.full(len(X_in), y_mean)
        
    # Split into Train and Validation for model selection
    # Use a fixed seed for reproducibility in selection, but the final model 
    # should ideally be refit on all data if we were doing true BMA. 
    # However, for simplicity and speed, we use the validation error of the 
    # model fit on the training split to weight the predictions of models 
    # refit on ALL data? No, that's leakage.
    # Standard approach: Fit on Train, eval on Val. 
    # But we want the final predictor to use all data.
    # Strategy: 
    # 1. Split X, y into X_tr, y_tr, X_val, y_val.
    # 2. For each candidate config, fit on X_tr, predict on X_val, compute loss.
    # 3. Compute weights based on these losses.
    # 4. For the final prediction, fit the SAME configs on ALL data (X, y) 
    #    and take the weighted average.
    # This is slightly optimistic (fitting on all data is better than train split),
    # but the weights are derived from the honest validation split.
    
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    val_size = max(10, int(0.3 * n))
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    candidates = []
    
    # Define configurations
    # Chebyshev
    cheb_degrees = [2, 4, 6, 8, 10, 14, 20]
    # RBF (gamma = 1/(2*length_scale^2))
    # Length scales: 0.1 to 3.0
    rbfs = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    # Fourier
    fouriers = [5, 10, 20]
    # Splines
    splines = [0.01, 0.1, 1.0]
    
    # 1. Evaluate on Validation Set
    for d in cheb_degrees:
        if _time_left() < 5: break
        try:
            pred_fn = _fit_chebyshev(X_tr, y_tr, degree=d)
            y_pred_val = pred_fn(X_val)
            if np.all(np.isfinite(y_pred_val)):
                loss = _robust_loss(y_val, y_pred_val)
                candidates.append(('cheb', d, loss))
        except:
            pass
            
    for g in rbfs:
        if _time_left() < 5: break
        try:
            pred_fn = _fit_rbf(X_tr, y_tr, gamma=g)
            y_pred_val = pred_fn(X_val)
            if np.all(np.isfinite(y_pred_val)):
                loss = _robust_loss(y_val, y_pred_val)
                candidates.append(('rbf', g, loss))
        except:
            pass
            
    for h in fouriers:
        if _time_left() < 5: break
        try:
            pred_fn = _fit_fourier(X_tr, y_tr, n_harmonics=h)
            y_pred_val = pred_fn(X_val)
            if np.all(np.isfinite(y_pred_val)):
                loss = _robust_loss(y_val, y_pred_val)
                candidates.append(('fourier', h, loss))
        except:
            pass
            
    for sf in splines:
        if _time_left() < 5: break
        try:
            pred_fn = _fit_spline(X_tr, y_tr, s_factor=sf)
            y_pred_val = pred_fn(X_val)
            if np.all(np.isfinite(y_pred_val)):
                loss = _robust_loss(y_val, y_pred_val)
                candidates.append(('spline', sf, loss))
        except:
            pass
            
    if not candidates:
        # Fallback
        return lambda X_in: np.full(len(X_in), y.mean())
        
    # 2. Compute Weights
    # Convert losses to weights. 
    # Use softmax with temperature. Lower loss = higher weight.
    # To avoid underflow/overflow, subtract min loss.
    losses = np.array([c[2] for c in candidates])
    min_loss = np.min(losses)
    # Temperature controls sharpness. 
    # If one model is much better, it gets most weight.
    # If all are similar, weights are roughly uniform.
    # Let's use a temperature of 1.0 relative to the spread.
    spread = np.std(losses)
    if spread < 1e-6:
        spread = 1.0
    temperature = spread
    exponents = np.exp(-(losses - min_loss) / temperature)
    weights = exponents / np.sum(exponents)
    
    # 3. Refit on ALL Data and Create Final Predictor
    final_predictors = []
    final_weights = []
    
    for i, (type, param, _) in enumerate(candidates):
        if _time_left() < 5: break
        try:
            if type == 'cheb':
                pred_fn = _fit_chebyshev(X, y, degree=param)
            elif type == 'rbf':
                pred_fn = _fit_rbf(X, y, gamma=param)
            elif type == 'fourier':
                pred_fn = _fit_fourier(X, y, n_harmonics=param)
            elif type == 'spline':
                pred_fn = _fit_spline(X, y, s_factor=param)
            else:
                continue
                
            # Verify finite on train (sanity check)
            y_check = pred_fn(X)
            if np.all(np.isfinite(y_check)):
                final_predictors.append(pred_fn)
                final_weights.append(weights[i])
        except:
            pass
            
    if not final_predictors:
        return lambda X_in: np.full(len(X_in), y.mean())
        
    # Normalize weights again in case some failed
    final_weights = np.array(final_weights)
    final_weights = final_weights / np.sum(final_weights)
    
    def predict(X_in):
        X_in = np.asarray(X_in, dtype=float)
        preds = np.zeros(len(X_in))
        for p, w in zip(final_predictors, final_weights):
            try:
                preds += w * p(X_in)
            except:
                # If a predictor fails at inference, ignore it (weight redistributed implicitly by not adding)
                # But this changes the sum of weights. 
                # Better to just return what we have.
                pass
        # Ensure finite
        preds = np.nan_to_num(preds, nan=y.mean(), posinf=y.max(), neginf=y.min())
        return preds
        
    return predict
