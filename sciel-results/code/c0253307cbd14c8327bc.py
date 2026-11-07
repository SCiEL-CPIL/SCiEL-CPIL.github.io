"""SEARCH: generate many candidate equations, fit each, keep the best."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def _fit_and_score(equation_fn, X_train, y_train, params_init=None):
    """Fit params with BFGS, return (mse, params) or (inf, None) on failure."""
    try:
        from scipy.optimize import minimize
        if params_init is None:
            params_init = np.ones(10)
        
        def loss(p):
            try:
                pred = equation_fn(X_train[0], X_train[1], X_train[2], X_train[3], p)
                if not np.all(np.isfinite(pred)):
                    return 1e30
                err = pred - y_train
                return float(np.mean(err**2))
            except Exception:
                return 1e30
        
        result = minimize(loss, params_init, method='BFGS',
                          options={'maxiter': 200, 'gtol': 1e-5})
        if not np.all(np.isfinite(result.x)):
            return np.inf, None
        pred = equation_fn(X_train[0], X_train[1], X_train[2], X_train[3], result.x)
        if not np.all(np.isfinite(pred)):
            return np.inf, None
        mse = float(np.mean((pred - y_train)**2))
        return mse, result.x
    except Exception:
        return np.inf, None

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural Change: Quadratic Polynomial Basis.
    Uses 10 parameters for: Intercept, 4 Linear, 4 Quadratic, 1 Interaction.
    This provides a smooth, differentiable landscape for BFGS and captures
    curvature and main interactions without the local minima issues of RBFs.
    """
    # Normalize inputs to [0, 1] range to improve numerical stability for BFGS
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 39.0
    t_n = (temp - 1.0) / 11.0
    p_n = (pH - 1.0) / 4.0

    # Construct the 10 basis functions
    # 1. Intercept
    # 2. b_n
    # 3. s_n
    # 4. t_n
    # 5. p_n
    # 6. b_n^2
    # 7. s_n^2
    # 8. t_n^2
    # 9. p_n^2
    # 10. b_n * s_n (Interaction between first two inputs)
    
    # Stack into a matrix of shape (n, 10)
    # Using np.column_stack for clarity and efficiency
    basis = np.column_stack((
        np.ones_like(b_n),
        b_n,
        s_n,
        t_n,
        p_n,
        b_n**2,
        s_n**2,
        t_n**2,
        p_n**2,
        b_n * s_n
    ))
    
    # Linear combination of basis functions
    pred = basis @ params
    
    return pred

# EVOLVE-BLOCK-END
