import numpy as np
import time
from scipy.optimize import minimize

def _softplus(x):
    """Numerically stable softplus: log(1+exp(x)) without overflow."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def equation(b, s, temp, pH, params):
    p = params
    
    # Parameters with stable softplus
    K_s      = _softplus(p[0]) + 1e-6
    T_opt    = p[1]
    T_width  = _softplus(p[2]) + 1e-3
    pH_opt   = p[3]
    pH_width = _softplus(p[4]) + 1e-3
    mu_max   = _softplus(p[5])
    b_slope  = p[6]
    b_shift  = p[7]
    mu_base  = p[8]
    s_inh    = _softplus(p[9]) + 1e-6  # Substrate inhibition strength (Haldane)
    
    # Normalizations to keep parameters O(1) for BFGS starting at 1
    # b: [1, 5] -> [0, 1]
    bn = (b - 1.0) / 4.0
    
    # Temperature: [1, 40]. Optimal ~37.
    tn = (temp - 20.0) / 10.0
    phn = (pH - 7.0) / 5.0
    
    # Monod with Haldane Inhibition
    sfac = s / (K_s + s + (s**2) / (s_inh + 1e-9))
    
    # Temperature Bell
    tfac = np.exp(-0.5 * ((tn - T_opt) / T_width)**2)
    
    # pH Bell
    pfac = np.exp(-0.5 * ((phn - pH_opt) / pH_width)**2)
    
    # Density Inhibition
    arg = np.clip(b_slope * (bn - b_shift), -50, 50)
    dfac = 1.0 / (1.0 + np.exp(arg))
    
    return mu_max * sfac * tfac * pfac * dfac + mu_base

def _objective(params, b, s, temp, pH, y):
    """MSE objective for fitting."""
    pred = equation(b, s, temp, pH, params)
    if not np.all(np.isfinite(pred)):
        return 1e30
    return np.mean((pred - y) ** 2)

def fit_params(b, s, temp, pH, y, deadline_margin=5.0):
    """Fit parameters with initial BFGS + final L-BFGS-B polish pass."""
    start_time = time.time()
    
    # Initial fit with BFGS as required by contract
    x0 = np.ones(10)
    res = minimize(_objective, x0, args=(b, s, temp, pH, y), method='BFGS')
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 110 - deadline_margin:
        return res.x
    
    # Final high-precision polish with L-BFGS-B (bounded, more robust)
    # Use tight bounds to keep parameters in reasonable ranges
    bounds = [
        (-10, 10),   # K_s (softplus input)
        (-2, 4),     # T_opt (normalized)
        (-5, 10),    # T_width (softplus input)
        (-2, 2),     # pH_opt (normalized)
        (-5, 10),    # pH_width (softplus input)
        (-5, 10),    # mu_max (softplus input)
        (-20, 20),   # b_slope
        (-1, 2),     # b_shift (normalized)
        (-1, 2),     # mu_base
        (-5, 15),    # s_inh (softplus input)
    ]
    
    res2 = minimize(_objective, res.x, args=(b, s, temp, pH, y), 
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    
    # Check deadline again
    elapsed = time.time() - start_time
    if elapsed > 110 - deadline_margin:
        return res.x  # Return safer initial result if close to deadline
    
    # Use the better result
    if res2.fun < res.fun:
        return res2.x
    return res.x
