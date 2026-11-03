import numpy as np
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _find_sidon_set(n, max_size=10):
    """
    Find a Sidon set of size as large as possible in [0, n-1].
    Uses a greedy approach with backtracking for small improvements.
    """
    # Greedy construction
    # Start with 0
    S = [0]
    used_sums = {0}
    
    for i in range(1, n):
        # Check if i can be added
        valid = True
        for s in S:
            if (s + i) in used_sums:
                valid = False
                break
        if (i + i) in used_sums:
            valid = False
        
        if valid:
            S.append(i)
            for s in S[:-1]:
                used_sums.add(s + i)
            used_sums.add(i + i)
            
            if len(S) >= max_size:
                break

    # Try to improve by swapping? Or just return greedy.
    # Greedy is often good enough for small N.
    # For N=50, greedy might get 7 or 8.
    return S

def _optimize_weights(indices):
    """
    Optimize weights for a given set of indices.
    We formulate this as minimizing the peak autoconvolution 
    subject to sum(h)=1, h>=0.
    Since we suspect equal weights are optimal for Sidon sets, 
    we start there and use SLSQP to refine.
    """
    K = len(indices)
    if K == 0:
        return np.zeros(N)
    
    h0 = np.zeros(N)
    for idx in indices:
        h0[idx] = 1.0 / K
        
    # If K is small, equal weights are likely optimal.
    # If we want to be safe, we can run a quick optimization.
    # But the directive suggests using convex optimization.
    # The problem min max(h*h) is convex.
    
    def objective(x):
        h = np.zeros(N)
        for i, idx in enumerate(indices):
            h[idx] = x[i]
        h_conv = np.convolve(h, h, mode='full')
        return np.max(h_conv)
        
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    bounds = [(0, None) for _ in range(K)]
    
    # Initial guess is equal weights
    x0 = np.ones(K) / K
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
        if res.success:
            h_opt = np.zeros(N)
            for i, idx in enumerate(indices):
                h_opt[idx] = res.x[i]
            h_opt /= np.sum(h_opt)
            return h_opt
    except Exception:
        pass
        
    return h0

def construct():
    # 1. Find a large Sidon set
    sidon_set = _find_sidon_set(N, 10)
    
    # 2. Optimize weights
    h_final = _optimize_weights(sidon_set)
        
    return list(h_final)
