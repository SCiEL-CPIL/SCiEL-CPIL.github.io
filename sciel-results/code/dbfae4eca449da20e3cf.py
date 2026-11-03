"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy import optimize

N = 50

def _score(h):
    """Compute the objective: (sum h^2)^2 / (sum h * max autoconv)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return -np.inf
    h = np.maximum(h, 0.0)
    s2 = np.sum(h**2)
    s1 = np.sum(h)
    if s1 < 1e-15 or s2 < 1e-15:
        return 0.0
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    if max_conv < 1e-15:
        return 0.0
    return (s2**2) / (s1 * max_conv)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try various structured patterns
    patterns = []
    
    # All ones (baseline)
    patterns.append(np.ones(N))
    
    # Single spike
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        patterns.append(h)
    
    # Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            patterns.append(h)
    
    # Ramp up
    h = np.linspace(0, 1, N)
    patterns.append(h)
    
    # Ramp down
    h = np.linspace(1, 0, N)
    patterns.append(h)
    
    # Triangle
    h = np.abs(np.linspace(-1, 1, N))
    patterns.append(h)
    
    # Cosine bell
    h = np.cos(np.pi * np.linspace(-0.5, 0.5, N))**2
    patterns.append(np.maximum(h, 0))
    
    # Gaussian-like
    x = np.linspace(-3, 3, N)
    h = np.exp(-x**2)
    patterns.append(h)
    
    # Block of ones in middle
    for width in range(1, N+1):
        start = (N - width) // 2
        h = np.zeros(N)
        h[start:start+width] = 1.0
        patterns.append(h)
    
    # Evaluate all patterns
    for h in patterns:
        if time.time() > deadline:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization from best starting point
    if time.time() < deadline:
        def neg_score(h):
            h = np.maximum(h, 0.0)
            return -_score(h)
        
        x0 = best_h.copy()
        try:
            res = optimize.minimize(neg_score, x0, method='Nelder-Mead', 
                                   options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                            'adaptive': True, 'maxfev': 5000})
            if time.time() < deadline:
                h_opt = np.maximum(res.x, 0.0)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    while time.time() < deadline - 5.0:
        # Random pattern
        h = np.random.exponential(1.0, size=N)
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        
        # Local optimize from this random point
        if time.time() < deadline - 10.0:
            def neg_score(h):
                h = np.maximum(h, 0.0)
                return -_score(h)
            try:
                res = optimize.minimize(neg_score, best_h.copy(), method='Nelder-Mead',
                                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                                'adaptive': True, 'maxfev': 1000})
                h_opt = np.maximum(res.x, 0.0)
                s_opt = _score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Normalize to ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
# EVOLVE-BLOCK-END
