import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_max_smooth(h, beta=100.0):
    """Smoothed max of autoconvolution using log-sum-exp."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m <= 0:
        return m
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    return m + np.log(np.sum(np.exp(shifted))) / beta

def _objective_smooth(coeffs, beta=100.0):
    """Objective: 2*N*max(autoconv) / sum(h)^2, with h constructed from coeffs."""
    # h = base + B @ coeffs, where base ensures sum(h) = 1
    # Use DCT-II basis for the variable part
    # coeffs[0] is the DC offset adjustment
    # Actually, let's parameterize: h = c0 * ones + B_var @ c_var
    # where B_var are DCT basis vectors (excluding DC), and we fix sum(h)=1
    
    c0 = coeffs[0]
    c_var = coeffs[1:]
    
    # Build h
    t = np.arange(N)
    # DCT-II basis vectors (orthonormal)
    # v_k[i] = sqrt(2/N) * cos(pi * (2i+1) * k / (2N)) for k=1,...,N-1
    # But constructing full B is expensive. Instead, use a small set of low-frequency modes.
    
    # For efficiency, use only first K modes
    K = min(len(c_var), 20)
    h = np.zeros(N)
    h += c0
    for k in range(1, K + 1):
        if k - 1 < len(c_var):
            mode = c_var[k - 1]
            h += mode * np.cos(np.pi * (2 * t + 1) * k / (2 * N))
    
    # Normalize to sum = 1
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h = h / s
    
    conv_max = _autoconv_max_smooth(h, beta)
    val = 2.0 * N * conv_max  # since sum(h) = 1
    return val

def _objective_hard(h):
    """Hard objective for final evaluation."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Seed solution
    h_best = np.ones(N)
    best_val = _objective_hard(h_best)  # Should be 2.0
    
    rng = np.random.default_rng(42)
    
    # Try structured optimization with different basis sizes and betas
    K_values = [10, 20, 30]
    beta_values = [50.0, 100.0, 200.0]
    
    for K in K_values:
        if time.time() > deadline - 30.0:
            break
        
        for beta in beta_values:
            if time.time() > deadline - 30.0:
                break
            
            # Initial coefficients: zero (gives constant h)
            x0 = np.zeros(K + 1)
            x0[0] = 1.0  # DC component
            
            try:
                res = minimize(
                    _objective_smooth,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x_opt = res.x
                
                # Reconstruct h
                t = np.arange(N)
                h = np.zeros(N)
                h += x_opt[0]
                c_var = x_opt[1:]
                for k in range(1, min(K, len(c_var)) + 1):
                    h += c_var[k - 1] * np.cos(np.pi * (2 * t + 1) * k / (2 * N))
                
                s = np.sum(h)
                if abs(s) < 1e-10:
                    continue
                h = h / s
                
                val = _objective_hard(h)
                if val < best_val:
                    best_val = val
                    h_best = h.copy()
                    
            except Exception:
                continue
    
    # Also try random restarts with local search
    if time.time() < deadline - 20.0:
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Random perturbation of best
            h = h_best + 0.1 * rng.standard_normal(N)
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            h = h / s
            
            # Local search
            for _ in range(50):
                if time.time() > deadline - 10.0:
                    break
                idx = rng.integers(0, N)
                delta = rng.normal(0, 0.01)
                h_new = h.copy()
                h_new[idx] += delta
                s_new = np.sum(h_new)
                if abs(s_new) < 1e-6:
                    continue
                h_new = h_new / s_new
                val_new = _objective_hard(h_new)
                val_curr = _objective_hard(h)
                if val_new < val_curr:
                    h = h_new
            
            val = _objective_hard(h)
            if val < best_val:
                best_val = val
                h_best = h.copy()
    
    return h_best
