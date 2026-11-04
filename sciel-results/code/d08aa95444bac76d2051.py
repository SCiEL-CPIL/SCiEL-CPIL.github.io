import numpy as np
from scipy.optimize import differential_evolution

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score(params):
    """
    params: [c1, c2, s1, s2, b]
    h = (w1 + b * w2)^2 where w1, w2 are Gaussians.
    w1 centered at c1 with width s1.
    w2 centered at c2 with width s2.
    We enforce c1 < c2 and separation.
    """
    c1, c2, s1, s2, b = params
    
    # Enforce order and separation
    if c1 > c2:
        c1, c2 = c2, c1
        s1, s2 = s2, s1
        
    # Enforce minimum separation to ensure disjoint supports
    # Gaussians decay, so we consider them effectively disjoint if centers are far
    # Let's say 3 sigma separation is enough for "disjoint" in this context
    min_sep = 3.0 * (s1 + s2)
    if c2 - c1 < min_sep:
        # Push them apart
        mid = (c1 + c2) / 2.0
        c1 = mid - min_sep / 2.0
        c2 = mid + min_sep / 2.0
        
    # Clip centers to domain
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    # If clipping caused overlap, force separation
    if c2 - c1 < min_sep:
         # This case is tricky if we hit boundaries. 
         # If we can't separate them within domain, the "disjoint" assumption fails.
         # But domain is [-0.25, 0.25], length 0.5.
         # If s1+s2 is large, we can't separate.
         # We should constrain s1, s2 to be small enough.
         pass

    s1 = max(s1, 0.001)
    s2 = max(s2, 0.001)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    f = w1 + b * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * dx
    if l1 <= 0:
        return -1.0
        
    l2_sq = np.sum(h**2) * dx
    
    conv = np.convolve(h, h, mode='full') * dx
    sup_norm = np.max(conv)
    
    if sup_norm <= 0:
        return -1.0
        
    return l2_sq / (l1 * sup_norm)

def _neg_score(params):
    s = _compute_score(params)
    if s < 0:
        return 1e10
    return -s

def construct():
    # Parameters: c1, c2, s1, s2, b
    # c1, c2: centers in [-0.25, 0.25]
    # s1, s2: widths (sigma)
    # b: relative amplitude
    
    bounds = [
        (-0.25, 0.0),   # c1 (left half)
        (0.0, 0.25),    # c2 (right half)
        (0.005, 0.05),  # s1
        (0.005, 0.05),  # s2
        (0.0, 1.0)      # b
    ]
    
    try:
        result = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=200, 
            popsize=30, 
            tol=1e-8, 
            seed=42,
            init='sobol'
        )
        best_params = result.x
        best_score = -result.fun
    except Exception as e:
        # Fallback
        np.random.seed(42)
        best_score = -np.inf
        best_params = None
        
        for _ in range(5000):
            c1 = np.random.uniform(-0.2, -0.05)
            c2 = np.random.uniform(0.05, 0.2)
            s1 = np.random.uniform(0.005, 0.03)
            s2 = np.random.uniform(0.005, 0.03)
            b = np.random.uniform(0.1, 1.0)
            params = [c1, c2, s1, s2, b]
            s = _compute_score(params)
            if s > best_score:
                best_score = s
                best_params = params
                
        if best_params is None:
            return [1.0] * N

    c1, c2, s1, s2, b = best_params
    
    # Re-apply constraints for final construction
    if c1 > c2:
        c1, c2 = c2, c1
        s1, s2 = s2, s1
        
    min_sep = 3.0 * (s1 + s2)
    if c2 - c1 < min_sep:
        mid = (c1 + c2) / 2.0
        c1 = mid - min_sep / 2.0
        c2 = mid + min_sep / 2.0
        
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    s1 = max(s1, 0.001)
    s2 = max(s2, 0.001)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    f = w1 + b * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
