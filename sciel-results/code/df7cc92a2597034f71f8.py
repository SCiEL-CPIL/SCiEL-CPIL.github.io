import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score(params):
    """
    params: [a, b, c1, c2, s1, s2]
    h = (a * w1 + b * w2)^2
    w1, w2 are Gaussian pulses centered at c1, c2 with widths s1, s2.
    Returns the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    a, b, c1, c2, s1, s2 = params
    
    # Ensure widths are positive and reasonable
    s1 = max(abs(s1), 0.001)
    s2 = max(abs(s2), 0.001)
    
    # Clip centers to domain to keep pulses relevant
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    # Compute Gaussian windows
    # Using exp(-0.5 * ((x-c)/s)^2)
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    # Sum of weighted windows
    f = a * w1 + b * w2
    
    # Squared modulus (which is just square since real)
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
    # Parameters: a, b, c1, c2, s1, s2
    # We will use a multi-start local optimization strategy.
    # The parameter space is small (6D).
    
    # Bounds for initial guesses and constraints
    # a, b: amplitudes. Scale invariance, but we bound to avoid 0.
    # c1, c2: centers.
    # s1, s2: widths.
    
    # To ensure disjointness, we encourage |c1 - c2| > 3*(s1+s2).
    # We will generate random starts that satisfy this roughly.
    
    np.random.seed(42)
    
    best_score = -np.inf
    best_params = None
    
    # Number of random starts
    n_starts = 50
    
    for i in range(n_starts):
        # Random widths
        s1 = np.random.uniform(0.005, 0.03)
        s2 = np.random.uniform(0.005, 0.03)
        
        # Random centers, ensuring separation
        # Domain is [-0.25, 0.25]. Total length 0.5.
        # Max separation is 0.5.
        # We want them far apart.
        # Pick c1, then pick c2 such that |c1-c2| is large.
        c1 = np.random.uniform(-0.2, 0.2)
        # Distance at least 3*(s1+s2)
        min_dist = 3.0 * (s1 + s2)
        
        # Try to place c2 far from c1
        # If c1 is near left, c2 near right, and vice versa.
        if c1 < 0:
            c2 = np.random.uniform(max(-0.25, c1 + min_dist), 0.25)
        else:
            c2 = np.random.uniform(-0.25, min(0.25, c1 - min_dist))
            
        # If the interval is invalid, adjust
        if c2 < c1 + min_dist and c2 > c1 - min_dist:
             # Fallback: just pick a random valid point
             valid_points = np.array([p for p in np.linspace(-0.25, 0.25, 100) if abs(p-c1) > min_dist])
             if len(valid_points) > 0:
                 c2 = valid_points[np.random.randint(len(valid_points))]
             else:
                 continue # Skip this start if no valid c2

        # Amplitudes
        a = np.random.uniform(0.5, 1.0)
        b = np.random.uniform(0.5, 1.0)
        
        x0 = [a, b, c1, c2, s1, s2]
        
        # Optimize locally
        # Use BFGS or Nelder-Mead
        # Nelder-Mead is robust for non-differentiable/noisy, but our function is smooth.
        # Let's use BFGS with gradient estimation or just Nelder-Mead for simplicity and robustness.
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 200})
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    # If local optimization failed or found nothing, fallback to DE or random
    if best_params is None:
        # Fallback to a quick DE
        bounds = [
            (0.1, 1.0),   # a
            (0.1, 1.0),   # b
            (-0.25, 0.25), # c1
            (-0.25, 0.25), # c2
            (0.005, 0.05),  # s1
            (0.005, 0.05)   # s2
        ]
        result = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6, 
            seed=42
        )
        best_params = result.x
        best_score = -result.fun

    a, b, c1, c2, s1, s2 = best_params
    s1 = max(abs(s1), 0.001)
    s2 = max(abs(s2), 0.001)
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    f = a * w1 + b * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
