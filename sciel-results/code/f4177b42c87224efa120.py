import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 195.0  # safety margin before 210s kill

def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound on C4 for the test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is chosen so g(0) = 0.
    
    The bound is ||x*g||_2 * ||g'||_2 / ||g||_2^2
    """
    c12 = -(c0 * 1.0 + c4 * 12.0 + c8 * 1152.0) / 460800.0
    
    x = np.linspace(-12, 12, 25001)
    dx = x[1] - x[0]
    gauss = np.exp(-0.5 * x**2)
    
    H0 = np.ones_like(x)
    H4 = 16.0 * x**4 - 48.0 * x**2 + 12.0
    H8 = 256.0 * x**8 - 3072.0 * x**6 + 11520.0 * x**4 - 11520.0 * x**2 + 1152.0
    H12 = (4096.0 * x**12 - 98304.0 * x**10 + 1048320.0 * x**8 
           - 6635520.0 * x**6 + 16588800.0 * x**4 - 18432000.0 * x**2 + 460800.0)
    
    P = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = P * gauss
    
    norm_g_sq = np.trapz(g**2, x)
    norm_xg_sq = np.trapz((x * g)**2, x)
    
    H4p = 64.0 * x**3 - 96.0 * x
    H8p = 2048.0 * x**7 - 18432.0 * x**5 + 46080.0 * x**3 - 23040.0 * x
    H12p = (49152.0 * x**11 - 983040.0 * x**9 + 8388608.0 * x**7 
            - 39813120.0 * x**5 + 66355200.0 * x**3 - 36864000.0 * x)
    
    Pp = c4 * H4p + c8 * H8p + c12 * H12p
    g_prime = (Pp - x * P) * gauss
    norm_gp_sq = np.trapz(g_prime**2, x)
    
    if norm_g_sq < 1e-30 or norm_xg_sq < 1e-30 or norm_gp_sq < 1e-30:
        return 1e10
    
    return np.sqrt(norm_xg_sq * norm_gp_sq) / norm_g_sq

def construct():
    t0 = time.time()
    
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    
    best_val = _compute_bound(c0_init, c4_init, c8_init)
    best_params = np.array([c0_init, c4_init, c8_init])
    
    def objective(params):
        if time.time() - t0 > _DEADLINE:
            return 1e10
        try:
            v = _compute_bound(params[0], params[1], params[2])
            return v if np.isfinite(v) else 1e10
        except Exception:
            return 1e10
    
    starts = [
        np.array([c0_init, c4_init, c8_init]),
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-1.5, 0.1, 0.001]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-2.0, 0.2, 0.005]),
        np.array([-0.8, -0.05, -0.001]),
        np.array([-1.3, 0.06, 0.002]),
        np.array([-1.0, 0.0, 0.0]),
    ]
    
    for start in starts:
        if time.time() - t0 > _DEADLINE - 15:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if time.time() - t0 < _DEADLINE - 20:
        try:
            res2 = minimize(objective, best_params, method='BFGS',
                           options={'maxiter': 1500, 'gtol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    if time.time() - t0 < _DEADLINE - 5:
        try:
            res3 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res3.fun < best_val:
                best_val = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
    
    return np.array([best_params[0], best_params[1], best_params[2]])
