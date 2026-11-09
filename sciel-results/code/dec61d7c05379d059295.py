"""Time-aware internal least-squares fit over a rich feature basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = None
_TIME_MARGIN = 5.0  # seconds safety margin before the 150s kill

def _elapsed_ok():
    """Return True if we still have time to safely finish."""
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _TIME_MARGIN

def _build_features(X, Y):
    """Build a diverse, bounded feature matrix from inputs."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]
    Xc = np.clip(X, -1.0, 1.0)
    Yc = np.clip(Y, -1.0, 1.0)
    
    # Radial and angular
    R = np.sqrt(Xc * Xc + Yc * Yc)
    theta = np.arctan2(Yc, Xc)
    
    feats = [np.ones(n)]
    # Linear
    feats += [Xc, Yc]
    # Quadratic
    feats += [Xc * Xc, Yc * Yc, Xc * Yc]
    # Cubic
    feats += [Xc**3, Yc**3, Xc**2 * Yc, Xc * Yc**2]
    # Trig
    feats += [np.sin(np.pi * Xc), np.cos(np.pi * Xc), np.sin(np.pi * Yc), np.cos(np.pi * Yc)]
    feats += [np.sin(2 * np.pi * Xc), np.cos(2 * np.pi * Xc), np.sin(2 * np.pi * Yc), np.cos(2 * np.pi * Yc)]
    # Radial
    feats += [R, R**2, R**3, np.sin(np.pi * R), np.cos(np.pi * R)]
    # Angular
    feats += [np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta)]
    # Interactions with R
    feats += [Xc * R, Yc * R]
    
    # Stack and normalize features to prevent numerical instability
    F = np.column_stack(feats)
    # Handle potential NaNs/Infs from trig/radial (though inputs are bounded)
    F = np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Standardize features (zero mean, unit variance) for better conditioning
    # But keep the intercept column as is
    F_std = F.copy()
    means = F_std[:, 1:].mean(axis=0)
    stds = F_std[:, 1:].std(axis=0)
    stds[stds == 0] = 1.0
    F_std[:, 1:] = (F_std[:, 1:] - means) / stds
    
    return F_std, means, stds

def _fit_params(F, y, initial_params):
    """
    Fit parameters using BFGS on the training data.
    Returns the best params found.
    """
    n = y.shape[0]
    
    def loss(params):
        # Ensure finite
        if not np.all(np.isfinite(params)):
            return 1e30
        pred = F @ params
        if not np.all(np.isfinite(pred)):
            return 1e30
        return np.mean((pred - y) ** 2)
    
    try:
        res = minimize(loss, initial_params, method='BFGS', options={'maxiter': 50})
        if res.success or res.fun < 1e29:
            return res.x
        else:
            return initial_params
    except Exception:
        return initial_params

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    
    # Build features
    F, means, stds = _build_features(x, y)
    num_feats = F.shape[1]
    
    # Initial parameters: all ones (as per task description start)
    # But we can be smarter: start with zeros for non-intercept, and mean(y) for intercept
    initial_params = np.zeros(num_feats)
    initial_params[0] = np.mean(y)
    
    # We have 10 params in the input, but our feature matrix has more.
    # The task says "params, an array of length 10".
    # This implies the evaluator passes a fixed-size array of 10 params.
    # We must map our internal fit to these 10 params?
    # No, the task says: "The evaluator fits params, an array of length 10... minimizing the training MSE."
    # And "equation(x, y, params) -> np.ndarray".
    # This means the *evaluator* calls minimize, which calls equation, which uses the *current* params.
    # We cannot change the number of params the evaluator uses.
    # We MUST use exactly the 10 params provided by the evaluator in the `params` argument.
    
    # RE-READ: "The evaluator fits params, an array of length 10... minimizing the training MSE."
    # This means the evaluator runs:
    # res = minimize(lambda p: mse(equation(x, y, p), y), initial_params=np.ones(10), method='BFGS')
    # Our function `equation` receives the current `params` (length 10) and must return predictions.
    # We CANNOT do our own fitting inside `equation` that changes the number of parameters, because the evaluator's optimizer is driving the 10 params.
    
    # So, the strategy "use equation as a black-box optimizer" is invalid because we don't control the optimizer loop. We only control the mapping from the 10 params to the prediction.
    
    # CORRECTED STRATEGY:
    # We have 10 free parameters. We need to design a model with exactly 10 coefficients that is expressive enough to fit the hidden function.
    # The current model is linear: p0*x + p1*y + p2. (3 params used, 7 unused).
    # We should use all 10 params for a richer model.
    
    # Let's design a model with 10 terms:
    # 1. Intercept
    # 2. x
    # 3. y
    # 4. x^2
    # 5. y^2
    # 6. x*y
    # 7. x^3
    # 8. y^3
    # 9. sin(pi*x)
    # 10. sin(pi*y)
    
    # This is a fixed basis. The evaluator will fit the 10 coefficients.
    # This is a different design family from the current linear model.
    
    X = x
    Y = y
    
    # Construct the 10 terms
    term1 = np.ones(n)
    term2 = X
    term3 = Y
    term4 = X * X
    term5 = Y * Y
    term6 = X * Y
    term7 = X * X * X
    term8 = Y * Y * Y
    term9 = np.sin(np.pi * X)
    term10 = np.sin(np.pi * Y)
    
    # Map params to these terms
    # params[0] -> term1
    # params[1] -> term2
    # ...
    # params[9] -> term10
    
    pred = (params[0] * term1 + 
            params[1] * term2 + 
            params[2] * term3 + 
            params[3] * term4 + 
            params[4] * term5 + 
            params[5] * term6 + 
            params[6] * term7 + 
            params[7] * term8 + 
            params[8] * term9 + 
            params[9] * term10)
            
    return pred
# EVOLVE-BLOCK-END
