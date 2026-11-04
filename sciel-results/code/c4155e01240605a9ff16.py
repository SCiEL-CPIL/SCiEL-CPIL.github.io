import numpy as np
from scipy.optimize import differential_evolution

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score(params):
    """
    params: [a1, a2, s1, s2, d, offset]
    Two disjoint Gaussian pulses:
      w1 centered at offset - d/2, width s1
      w2 centered at offset + d/2, width s2
    h = (a1*w1 + a2*w2)^2
    Returns ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    a1, a2, s1, s2, d, offset = params
    
    s1 = max(abs(s1), 0.002)
    s2 = max(abs(s2), 0.002)
    
    c1 = offset - d / 2.0
    c2 = offset + d / 2.0
    
    c1 = np.clip(c1, -0.24, 0.24)
    c2 = np.clip(c2, -0.24, 0.24)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    f = a1 * w1 + a2 * w2
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
    bounds = [
        (0.1, 1.0),    # a1
        (0.1, 1.0),    # a2
        (0.003, 0.08), # s1
        (0.003, 0.08), # s2
        (0.05, 0.45),  # d
        (-0.1, 0.1)    # offset
    ]
    
    try:
        result = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=500, 
            popsize=30, 
            tol=1e-10, 
            seed=42,
            init='sobol',
            polish=True
        )
        best_params = result.x
        best_score = -result.fun
    except Exception:
        np.random.seed(42)
        best_score = -np.inf
        best_params = None
        
        for _ in range(5000):
            a1 = np.random.uniform(0.3, 1.0)
            a2 = np.random.uniform(0.3, 1.0)
            s1 = np.random.uniform(0.005, 0.06)
            s2 = np.random.uniform(0.005, 0.06)
            d = np.random.uniform(0.15, 0.45)
            offset = np.random.uniform(-0.05, 0.05)
            params = [a1, a2, s1, s2, d, offset]
            s = _compute_score(params)
            if s > best_score:
                best_score = s
                best_params = params
        
        if best_params is None:
            return [1.0] * N

    a1, a2, s1, s2, d, offset = best_params
    s1 = max(abs(s1), 0.002)
    s2 = max(abs(s2), 0.002)
    
    c1 = offset - d / 2.0
    c2 = offset + d / 2.0
    c1 = np.clip(c1, -0.24, 0.24)
    c2 = np.clip(c2, -0.24, 0.24)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    f = a1 * w1 + a2 * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
