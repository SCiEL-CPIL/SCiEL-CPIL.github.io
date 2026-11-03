"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(h):
    """Compute the ratio ||h||^2_2 / (||h||_1 * ||h*h||_inf) for linear autoconvolution."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -np.inf
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup = np.max(conv)
    if sup < 1e-15:
        return -np.inf
    return l2_sq / (l1 * sup)

def _score_circular(h):
    """Compute the ratio with circular autoconvolution."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -np.inf
    # Circular autoconvolution via FFT
    n = len(h)
    H = np.fft.fft(h)
    conv = np.real(np.fft.ifft(H * H))
    sup = np.max(conv)
    if sup < 1e-15:
        return -inf
    return l2_sq / (l1 * sup)

def _neg_score_linear(h):
    return -_score_linear(h)

def _score_linear(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -np.inf
    conv = np.convolve(h, h, mode='full')
    sup = np.max(conv)
    if sup < 1e-15:
        return -np.inf
    return l2_sq / (l1 * sup)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _score_linear(best_h)
    
    # Try delta
    delta = np.zeros(N)
    delta[0] = 1.0
    val_delta = _score_linear(delta)
    if val_delta > best_val:
        best_h = delta.copy()
        best_val = val_delta
    
    # Try various structured initializations and optimize
    # Use scipy to maximize the ratio. Since the ratio is scale-invariant in a specific way,
    # let's normalize ||h||_1 = 1 and maximize ||h||^2_2 / ||h*h||_inf
    
    def _objective_normalized(x):
        """Minimize negative of ||h||^2_2 / ||h*h||_inf subject to sum(h)=1, h>=0."""
        h = np.maximum(x, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s  # normalize to L1 = 1
        l2_sq = np.sum(h ** 2)
        conv = np.convolve(h, h, mode='full')
        sup = np.max(conv)
        if sup < 1e-15:
            return 1e10
        return -l2_sq / sup
    
    # Try multiple random starts
    np.random.seed(42)
    best_opt_h = best_h.copy()
    best_opt_val = best_val
    
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        # Different random initializations
        if i % 5 == 0:
            # Random
            x0 = np.random.rand(N)
        elif i % 5 == 1:
            # Gaussian-like
            x0 = np.exp(-((np.arange(N) - N//2)**2) / (2 * (N//4)**2))
        elif i % 5 == 2:
            # Triangle
            x0 = np.abs(np.arange(N) - N//2)
            x0 = N//2 - x0
            x0 = np.maximum(x0, 0)
        elif i % 5 == 3:
            # Two spikes
            x0 = np.zeros(N)
            x0[N//4] = 1
            x0[3*N//4] = 1
        else:
            # Sparse random
            x0 = np.zeros(N)
            idx = np.random.choice(N, size=max(1, N//10), replace=False)
            x0[idx] = 1.0
        
        try:
            res = minimize(_objective_normalized, x0, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'maxfun': 2000})
            if res.success or res.fun < 1e9:
                h_opt = np.maximum(res.x, 0.0)
                s = np.sum(h_opt)
                if s > 1e-15:
                    h_opt = h_opt / s
                    val = _score_linear(h_opt)
                    if val > best_opt_val:
                        best_opt_val = val
                        best_opt_h = h_opt.copy()
        except Exception:
            pass
    
    if best_opt_val > best_val:
        best_h = best_opt_h
        best_val = best_opt_val
    
    # Also try with circular convolution
    def _objective_circ(x):
        h = np.maximum(x, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        l2_sq = np.sum(h ** 2)
        H = np.fft.fft(h)
        conv = np.real(np.fft.ifft(H * H))
        sup = np.max(conv)
        if sup < 1e-15:
            return 1e10
        return -l2_sq / sup
    
    for i in range(20):
        if time.time() > deadline:
            break
        x0 = np.random.rand(N)
        try:
            res = minimize(_objective_circ, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'maxfun': 1000})
            if res.success or res.fun < 1e9:
                h_opt = np.maximum(res.x, 0.0)
                s = np.sum(h_opt)
                if s > 1e-15:
                    h_opt = h_opt / s
                    # Score with linear conv (the actual metric)
                    val = _score_linear(h_opt)
                    if val > best_val:
                        best_val = val
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
