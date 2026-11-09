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
    s_inh    = _softplus(p[9]) + 1e-6 # Substrate inhibition strength (Haldane)
    
    # Normalizations to keep parameters O(1) for BFGS starting at 1
    # b: [1, 5] -> [0, 1]
    bn = (b - 1.0) / 4.0
    tn = (temp - 20.0) / 10.0
    phn = (pH - 7.0) / 5.0
    
    # Monod with Haldane Inhibition
    # Use log-sum-exp style stable computation if possible, but standard is fine with clipping
    # s is in [1, 40], K_s ~ 1, s_inh ~ 10
    # s^2 / s_inh can be up to 1600/10 = 160.
    # Denominator: K_s + s + s^2/s_inh. Max ~ 1 + 40 + 160 = 201.
    # This is safe.
    sfac = s / (K_s + s + (s**2) / (s_inh + 1e-9))
    
    # Temperature Bell
    # tn ~ [0, 1], T_opt ~ 1.7 (37C), T_width ~ 0.5
    # ((tn - T_opt) / T_width)^2 can be large if T_opt is far.
    # Clip exponent to avoid underflow/overflow in exp (though exp(-large) is just 0)
    t_arg = np.clip(-0.5 * ((tn - T_opt) / T_width)**2, -100, 0)
    tfac = np.exp(t_arg)
    
    # pH Bell
    # phn ~ [0, 1], pH_opt ~ 0, pH_width ~ 1
    p_arg = np.clip(-0.5 * ((phn - pH_opt) / pH_width)**2, -100, 0)
    pfac = np.exp(p_arg)
    
    # Density Inhibition
    arg = np.clip(b_slope * (bn - b_shift), -50, 50)
    dfac = 1.0 / (1.0 + np.exp(arg))
    
    return mu_max * sfac * tfac * pfac * dfac + mu_base

def _objective(params, b, s, temp, pH, y):
    try:
        pred = equation(b, s, temp, pH, params)
        if not np.all(np.isfinite(pred)):
            return 1e30
        return np.mean((pred - y)**2)
    except Exception:
        return 1e30

def _gradient(params, b, s, temp, pH, y):
    """
    Numerical gradient via central differences.
    BFGS can be slow if it estimates its own gradient poorly or if the landscape is stiff.
    Providing an exact (numerical) gradient helps BFGS converge faster and more reliably.
    """
    eps = 1e-5
    n = len(params)
    grad = np.zeros(n)
    f0 = _objective(params, b, s, temp, pH, y)
    
    if not np.isfinite(f0):
        return grad # Zero gradient if objective is invalid, BFGS will handle it or fail
    
    for i in range(n):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus, b, s, temp, pH, y)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _objective(params_minus, b, s, temp, pH, y)
        
        if np.isfinite(f_plus) and np.isfinite(f_minus):
            grad[i] = (f_plus - f_minus) / (2 * eps)
        elif np.isfinite(f_plus):
            grad[i] = (f_plus - f0) / eps
        elif np.isfinite(f_minus):
            grad[i] = (f0 - f_minus) / eps
        else:
            grad[i] = 0.0
            
    return grad

def fit_params(b, s, temp, pH, y, timeout=140.0):
    start_time = time.time()
    deadline = start_time + timeout
    
    # Initial guesses
    # K_s ~ 1, T_opt ~ 1.7 (37C), T_width ~ 0.5, pH_opt ~ 0, pH_width ~ 1, mu_max ~ 1.5, b_slope ~ 1, b_shift ~ 0.5, mu_base ~ 0, s_inh ~ 10
    x0 = np.array([0.0, 1.7, 0.5, 0.0, 1.0, 0.4, 1.0, 0.5, 0.0, 2.3])
    
    best_x = x0
    best_f = _objective(x0, b, s, temp, pH, y)
    
    # Try a few random restarts if time permits, otherwise just BFGS from x0
    # Since we have 150s limit, and BFGS is fast, we can try 2-3 restarts.
    
    restarts = [
        x0,
        x0 + np.array([0.5, 0.1, 0.2, 0.1, 0.2, 0.2, 0.5, 0.2, 0.0, 0.5]),
        x0 + np.array([-0.5, -0.1, -0.2, -0.1, -0.2, -0.2, -0.5, -0.2, 0.0, -0.5])
    ]
    
    for x_init in restarts:
        if time.time() > deadline - 10: # Safety margin
            break
            
        try:
            res = minimize(_objective, x_init, args=(b, s, temp, pH, y), method='BFGS', 
                           jac=_gradient,
                           options={'maxiter': 200, 'gtol': 1e-5})
            
            if res.success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish with tighter tolerance if time remains
    # Use L-BFGS-B or BFGS with high precision? 
    # BFGS with numerical gradient is robust. Let's do one more pass.
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_x, args=(b, s, temp, pH, y), method='BFGS', 
                           jac=_gradient,
                           options={'maxiter': 500, 'gtol': 1e-7})
            if res.success and res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass
            
    return best_x
