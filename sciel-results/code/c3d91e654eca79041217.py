import numpy as np
from scipy.optimize import minimize
import time

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
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Parameters: a, b, c1, c2, s1, s2
    # We will use multi-start Powell optimization
    # Powell is derivative-free and works well with bounds via transformation or penalties.
    # However, standard Powell doesn't support bounds directly. 
    # We can use Nelder-Mead with a penalty for out-of-bounds, or use minimize with 'L-BFGS-B' if we relax the non-negativity of a,b,s.
    # Since a,b,s > 0, we can parameterize them as exp(y) to ensure positivity.
    # c1, c2 are in [-0.25, 0.25].
    
    # Let's stick to a robust approach: Random initialization + Local Refinement (Nelder-Mead)
    # We need to handle bounds in Nelder-Mead by penalizing violations heavily.
    
    def _penalized_neg_score(params):
        a, b, c1, c2, s1, s2 = params
        
        # Penalty for bounds
        penalty = 0.0
        if a < 0: penalty += 1e10 * (a**2)
        if b < 0: penalty += 1e10 * (b**2)
        if s1 < 0.001: penalty += 1e10 * ((0.001 - s1)**2)
        if s2 < 0.001: penalty += 1e10 * ((0.001 - s2)**2)
        if c1 < -0.25: penalty += 1e10 * ((-0.25 - c1)**2)
        if c1 > 0.25: penalty += 1e10 * ((c1 - 0.25)**2)
        if c2 < -0.25: penalty += 1e10 * ((-0.25 - c2)**2)
        if c2 > 0.25: penalty += 1e10 * ((c2 - 0.25)**2)
        
        if penalty > 0:
            return 1e10 + penalty
            
        s = _compute_score(params)
        if s < 0:
            return 1e10
        return -s

    best_score = -np.inf
    best_params = None
    
    np.random.seed(42)
    
    # Define bounds for random initialization to encourage disjoint supports
    # c1 in [-0.25, 0], c2 in [0, 0.25] to force separation initially
    # Or just random and let the optimizer fix it.
    
    num_starts = 100
    max_iter_per_start = 500
    
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate initial guess
        # Encourage separation: c1 < c2
        c1 = np.random.uniform(-0.25, -0.05)
        c2 = np.random.uniform(0.05, 0.25)
        
        # Random widths
        s1 = np.random.uniform(0.01, 0.05)
        s2 = np.random.uniform(0.01, 0.05)
        
        # Random amplitudes
        a = np.random.uniform(0.5, 1.0)
        b = np.random.uniform(0.5, 1.0)
        
        x0 = np.array([a, b, c1, c2, s1, s2])
        
        try:
            res = minimize(
                _penalized_neg_score, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': max_iter_per_start, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        # Fallback
        return [1.0] * N

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
