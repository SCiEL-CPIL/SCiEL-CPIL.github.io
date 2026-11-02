import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def _smooth_max(x, beta):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    if beta * 0 > 700:
        return m
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    # IDCT-based parametrization:
    # h[i] = (1/N) * (c0 + 2*sum_{k=1}^{K-1} c_k * cos(pi*k*(2i+1)/(2N)))
    # This is the IDCT-II of c. For h >= 0, we need c to be in a cone.
    # We'll use a penalty approach: minimize smooth_max(conv) + penalty * max(0, -min(h))^2
    # with the constraint that h is normalized.
    
    def idct_to_h(c, K):
        """Convert spectral coefficients to time-domain h via IDCT-II, normalized to sum=1."""
        n = N
        # IDCT-II: h[i] = sum_{k=0}^{K-1} c[k] * cos(pi*k*(2*i+1)/(2*n))
        # We'll compute this efficiently
        i = np.arange(n)
        k = np.arange(K)
        # h[i] = sum_k c[k] * cos(pi*k*(2*i+1)/(2*n))
        # For efficiency, use the DCT matrix
        angles = np.pi * np.outer(i, k) * (2 * i + 1) / (2 * n)
        # Actually, let's just compute directly
        angles = np.pi * np.outer(k, (2 * i + 1)) / (2 * n)
        cos_matrix = np.cos(angles)  # shape (K, n)
        h = cos_matrix.T @ c  # shape (n,)
        
        # Ensure nonnegativity: clip negative values
        h = np.maximum(h, 0)
        
        s = np.sum(h)
        if s < 1e-15:
            return None
        h = h / s
        return h
    
    def objective_idct(c, K, beta, penalty_weight):
        h = idct_to_h(c, K)
        if h is None:
            return 1.0
        conv = np.convolve(h, h, mode='full')
        sm = _smooth_max(conv, beta)
        
        # Penalty for negative values before clipping (to encourage feasibility)
        i = np.arange(N)
        k = np.arange(K)
        angles = np.pi * np.outer(k, (2 * i + 1)) / (2 * N)
        cos_matrix = np.cos(angles)
        h_raw = cos_matrix.T @ c
        neg_penalty = penalty_weight * np.sum(np.maximum(-h_raw, 0) ** 2)
        
        return sm + neg_penalty
    
    # Try multiple K values (number of cosine coefficients)
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    for K in [2, 3, 5, 8, 12]:
        if time.time() > deadline - 15:
            break
        
        # Initial guess: uniform-like spectrum
        c0 = np.zeros(K)
        c0[0] = 1.0  # DC component dominates for uniform
        
        beta_schedule = [20.0, 50.0, 100.0]
        penalty_schedule = [1.0, 0.1, 0.01]
        
        for beta, pw in zip(beta_schedule, penalty_schedule):
            if time.time() > deadline - 10:
                break
            
            def obj(c):
                return objective_idct(c, K, beta, pw)
            
            try:
                res = minimize(obj, c0, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
                h_opt = idct_to_h(res.x, K)
                if h_opt is not None:
                    score = compute_score(h_opt)
                    if score < best_score:
                        best_score = score
                        best_h = h_opt.copy()
                        c0 = res.x.copy()
            except Exception:
                pass
        
        # Also try random initializations
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            c_rand = np.random.randn(K) * 0.1
            c_rand[0] = 1.0  # keep DC positive
            
            beta = 50.0
            pw = 0.1
            
            def obj(c):
                return objective_idct(c, K, beta, pw)
            
            try:
                res = minimize(obj, c_rand, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True})
                h_opt = idct_to_h(res.x, K)
                if h_opt is not None:
                    score = compute_score(h_opt)
                    if score < best_score:
                        best_score = score
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Additional approach: direct optimization with softplus transformation
    if time.time() < deadline - 20:
        def softplus(x):
            return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        
        def obj_direct(x, beta):
            h = softplus(x)
            s = np.sum(h)
            if s < 1e-15:
                return 1.0
            h = h / s
            conv = np.convolve(h, h, mode='full')
            return _smooth_max(conv, beta)
        
        # Start from uniform
        x0 = np.full(N, np.log(1.0 / N))
        
        for beta in [30.0, 60.0]:
            if time.time() > deadline - 15:
                break
            
            try:
                res = minimize(lambda x: obj_direct(x, beta), x0, method='L-BFGS-B',
                              options={'maxiter': 100, 'gtol': 1e-8})
                h_opt = softplus(res.x)
                s = np.sum(h_opt)
                if s > 1e-15:
                    h_opt = h_opt / s
                    score = compute_score(h_opt)
                    if score < best_score:
                        best_score = score
                        best_h = h_opt.copy()
                        x0 = res.x.copy()
            except Exception:
                pass
    
    # Final verification and cleanup
    if np.any(best_h < 0):
        best_h = np.maximum(best_h, 0)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.ones(N) / N
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
