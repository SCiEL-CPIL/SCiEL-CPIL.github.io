import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 600.0
    
    m = 12
    
    # Initial guess from Laguerre roots with alpha=14
    try:
        roots_lag, _ = roots_laguerre(m, 14.0)
        x0 = np.sort(roots_lag)
    except:
        x0 = np.linspace(4.0, 48.0, m)
    
    # Precompute quadrature for norm evaluation
    n_quad = 300
    x_q, w_q = roots_laguerre(n_quad, 0.0)
    
    def compute_log_norm(r):
        # P(x) = prod_{i=1}^{m} (x - r_i)^2, norm under e^{-x}
        log_P_sq = np.sum(2.0 * np.log(np.abs(x_q[:, None] - r[None, :])), axis=1)
        log_terms = log_P_sq + np.log(w_q)
        max_log = np.max(log_terms)
        return max_log + np.log(np.sum(np.exp(log_terms - max_log)))
    
    def objective(r):
        if np.any(r <= 0) or np.any(np.diff(r) < 1e-6):
            return 1e10
        return compute_log_norm(r)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                         bounds=[(0.1, 500.0)] * m,
                         options={'maxiter': 3000, 'ftol': 1e-14})
        
        if time.time() - start_time > deadline - 30:
            raise TimeoutError()
        
        r_opt = np.sort(result.x)
        r_opt = np.maximum(r_opt, 0.1)
        
        if time.time() - start_time > deadline - 10:
            return [4.0 * (i + 1) for i in range(m)]
            
        return list(r_opt)
    
    except Exception:
        return [4.0 * (i + 1) for i in range(m)]
