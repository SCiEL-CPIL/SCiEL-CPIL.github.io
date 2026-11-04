import numpy as np
from scipy.optimize import minimize

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score(params):
    """
    params: [a, b, d, w1, w2]
    h = (a * w1_pulse + b * w2_pulse)^2
    w1_pulse centered at -d/2, w2_pulse centered at d/2.
    """
    a, b, d, w1, w2 = params
    
    # Enforce positive widths
    w1 = max(abs(w1), 0.005)
    w2 = max(abs(w2), 0.005)
    
    # Centers
    c1 = -d / 2.0
    c2 = d / 2.0
    
    # Compute Gaussian windows
    # exp(-0.5 * ((x-c)/w)^2)
    w1_vals = np.exp(-0.5 * ((x - c1) / w1) ** 2)
    w2_vals = np.exp(-0.5 * ((x - c2) / w2) ** 2)
    
    # Sum of weighted windows
    f = a * w1_vals + b * w2_vals
    
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
        
    return l2_sq / (l1 * sup_norm)

def _neg_score(params):
    s = _compute_score(params)
    if s < 0:
        return 1e10
    return -s

def construct():
    # Parameters: a, b, d, w1, w2
    # a, b: amplitudes [0, 1]
    # d: separation [0.05, 0.5]
    # w1, w2: widths [0.005, 0.1]
    
    bounds = [
        (0.0, 1.0),   # a
        (0.0, 1.0),   # b
        (0.05, 0.5),  # d (separation)
        (0.005, 0.1), # w1
        (0.005, 0.1)  # w2
    ]
    
    # Multi-start Powell optimization
    best_score = -np.inf
    best_params = None
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    
    # Number of random starts
    n_starts = 20
    
    for i in range(n_starts):
        # Random initialization within bounds
        init_params = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1]),
            np.random.uniform(bounds[3][0], bounds[3][1]),
            np.random.uniform(bounds[4][0], bounds[4][1])
        ])
        
        # Ensure initial separation is reasonable to avoid singularities or bad starts
        # Powell is robust but let's just run it
        
        try:
            res = minimize(
                _neg_score, 
                init_params, 
                method='Powell', 
                bounds=bounds,
                options={'maxiter': 200, 'maxfev': 1000}
            )
            
            if res.success or res.fun < 1e9:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            continue
    
    # Fallback if optimization failed or didn't improve
    if best_params is None or best_score <= 0:
        # Try a few random evaluations
        for _ in range(100):
            params = np.array([
                np.random.uniform(0.0, 1.0),
                np.random.uniform(0.0, 1.0),
                np.random.uniform(0.05, 0.5),
                np.random.uniform(0.005, 0.1),
                np.random.uniform(0.005, 0.1)
            ])
            s = _compute_score(params)
            if s > best_score:
                best_score = s
                best_params = params
        
        if best_params is None:
            return [1.0] * N

    a, b, d, w1, w2 = best_params
    w1 = max(abs(w1), 0.005)
    w2 = max(abs(w2), 0.005)
    d = max(abs(d), 0.05)
    
    c1 = -d / 2.0
    c2 = d / 2.0
    
    w1_vals = np.exp(-0.5 * ((x - c1) / w1) ** 2)
    w2_vals = np.exp(-0.5 * ((x - c2) / w2) ** 2)
    
    f = a * w1_vals + b * w2_vals
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
