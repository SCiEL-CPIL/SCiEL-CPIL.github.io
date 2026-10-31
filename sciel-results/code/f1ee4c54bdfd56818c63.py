import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s

def construct():
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def compute_ratio(h):
        """Compute ||h||_2^2 / ||h*h||_inf, assuming h is L1-normalized."""
        h = np.asarray(h, dtype=np.float64)
        h2 = np.sum(h * h)
        # Autoconvolution
        hh = np.convolve(h, h, mode='full')
        hh_max = np.max(hh)
        if hh_max < 1e-300:
            return 0.0
        return h2 / hh_max
    
    def soft_ratio(h, alpha=100.0):
        """Smooth approximation of the ratio using log-sum-exp for max."""
        h = np.asarray(h, dtype=np.float64)
        h2 = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        # Soft max
        hh_max = np.log(np.sum(np.exp(alpha * hh))) / alpha
        if hh_max < 1e-300:
            return 0.0
        return h2 / hh_max
    
    def neg_soft_ratio(params, alpha=100.0):
        """Negative of soft ratio for minimization. params are log-space to ensure nonnegativity."""
        h = np.exp(params)
        h_norm = h / np.sum(h)  # L1 normalize
        return -soft_ratio(h_norm, alpha)
    
    # Strategy: optimize in log-space, L1-normalize
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    # Try several random initializations
    n_attempts = 0
    max_attempts = 50
    
    while elapsed() < DEADLINE and n_attempts < max_attempts:
        # Random initialization in log-space
        log_h = np.random.randn(N) * 2.0
        # Also try structured initializations
        if n_attempts == 0:
            # Single spike
            h_init = np.zeros(N)
            h_init[N//2] = 1.0
            best_h = h_init.copy()
            best_val = compute_ratio(h_init)
        
        if n_attempts == 1:
            # Two equal spikes
            h_init = np.zeros(N)
            h_init[N//4] = 0.5
            h_init[3*N//4] = 0.5
            val = compute_ratio(h_init)
            if val > best_val:
                best_val = val
                best_h = h_init.copy()
        
        if n_attempts == 2:
            # Many equal spikes
            h_init = np.zeros(N)
            positions = np.linspace(0, N-1, 10).astype(int)
            h_init[positions] = 1.0 / len(positions)
            val = compute_ratio(h_init)
            if val > best_val:
                best_val = val
                best_h = h_init.copy()
        
        if elapsed() > DEADLINE:
            break
        
        # Optimize from this initialization
        x0 = np.log(best_h + 1e-10) if n_attempts >= 3 else log_h
        # Actually, let's optimize from random starts after the structured ones
        if n_attempts >= 3:
            x0 = np.random.randn(N) * 2.0
        
        try:
            res = minimize(neg_soft_ratio, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12})
            if elapsed() > DEADLINE:
                break
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
        
        n_attempts += 1
    
    # Final polish: use exact ratio with more iterations from best
    if elapsed() < DEADLINE - 5:
        try:
            x0 = np.log(best_h + 1e-10)
            res = minimize(neg_soft_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15})
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)

# EVOLVE-BLOCK-END
