import numpy as np
from scipy.optimize import differential_evolution

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _hann_window(x, center, width):
    """
    Compactly supported Hann window.
    Non-zero only in [center - width/2, center + width/2].
    """
    w = np.zeros_like(x)
    half_w = width / 2.0
    mask = (x >= center - half_w) & (x <= center + half_w)
    if np.any(mask):
        t = (x[mask] - center) / half_w
        # Hann window: 0.5 * (1 + cos(pi * t))
        w[mask] = 0.5 * (1 + np.cos(np.pi * t))
    return w

def _compute_score(params):
    """
    params: [a, b, c1, c2, s1, s2]
    h = (a * w1 + b * w2)^2
    w1, w2 are compactly supported Hann windows.
    Returns the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    a, b, c1, c2, s1, s2 = params
    
    # Ensure widths are positive and reasonable
    s1 = max(abs(s1), 0.005)
    s2 = max(abs(s2), 0.005)
    
    # Clip centers to domain
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    # Compute compactly supported windows
    w1 = _hann_window(x, c1, s1)
    w2 = _hann_window(x, c2, s2)
    
    # Sum of weighted windows
    f = a * w1 + b * w2
    
    # Squared modulus
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * dx
    if l1 <= 0:
        return -1.0
        
    l2_sq = np.sum(h**2) * dx
    
    # Autoconvolution
    conv = np.convolve(h, h, mode='full') * dx
    sup_norm = np.max(conv)
    
    if sup_norm <= 0:
        return -1.0
        
    score = l2_sq / (l1 * sup_norm)
    
    # Penalty for overlapping supports
    # Distance between centers
    dist = abs(c1 - c2)
    min_dist = (s1 + s2) / 2.0
    if dist < min_dist:
        # Strong penalty to encourage disjointness
        penalty = (min_dist - dist) * 100.0
        score -= penalty
        
    return score

def _neg_score(params):
    s = _compute_score(params)
    if s < 0:
        return 1e10
    return -s

def construct():
    # Parameters: a, b, c1, c2, s1, s2
    # a, b: amplitudes
    # c1, c2: centers
    # s1, s2: widths
    
    bounds = [
        (0.0, 1.0),   # a
        (0.0, 1.0),   # b
        (-0.25, 0.25), # c1
        (-0.25, 0.25), # c2
        (0.01, 0.2),   # s1 (width)
        (0.01, 0.2)    # s2 (width)
    ]
    
    try:
        result = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=500, 
            popsize=30, 
            tol=1e-8, 
            seed=42,
            init='sobol',
            mutation=(0.5, 1.5),
            recombination=0.8
        )
        best_params = result.x
        best_score = -result.fun
    except Exception as e:
        # Fallback: Random search
        np.random.seed(42)
        best_score = -np.inf
        best_params = None
        
        for _ in range(3000):
            a = np.random.uniform(0.1, 1.0)
            b = np.random.uniform(0.1, 1.0)
            c1 = np.random.uniform(-0.2, 0.2)
            c2 = np.random.uniform(-0.2, 0.2)
            # Ensure separation
            if abs(c1 - c2) < 0.1:
                c2 = c1 + np.sign(c2-c1 if c2!=c1 else 1) * 0.15
            s1 = np.random.uniform(0.02, 0.1)
            s2 = np.random.uniform(0.02, 0.1)
            params = [a, b, c1, c2, s1, s2]
            s = _compute_score(params)
            if s > best_score:
                best_score = s
                best_params = params
        
        if best_params is None:
            return [1.0] * N

    a, b, c1, c2, s1, s2 = best_params
    s1 = max(abs(s1), 0.005)
    s2 = max(abs(s2), 0.005)
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    w1 = _hann_window(x, c1, s1)
    w2 = _hann_window(x, c2, s2)
    
    f = a * w1 + b * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
