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

def construct():
    # Initial guess: Uniform distribution
    h0 = np.ones(N) / N
    
    # Objective: Minimize negative log score to handle positivity and scale
    # Score = L2^2 / (L1 * MaxConv)
    # We fix L1=1 in constraints, so we maximize L2^2 / MaxConv
    # Maximize log(L2^2 / MaxConv) = 2*log(L2) - log(MaxConv)
    # Minimize -2*log(L2) + log(MaxConv)
    
    def objective(x):
        h = np.abs(x) # Enforce non-negativity via abs? No, use bounds.
        # x is h, constrained to be >= 0 and sum to 1
        l2_sq = np.sum(h**2)
        if l2_sq == 0:
            return 1e10
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return -1e10
        # We want to maximize l2_sq / max_conv
        # So minimize -log(l2_sq / max_conv)
        return -np.log(l2_sq) + np.log(max_conv)
        
    def gradient(x):
        # Analytical gradient is complex due to max. Use numerical or SLSQP internal.
        # SLSQP can handle this with numerical gradients or we provide approx.
        # Let's rely on SLSQP's numerical gradient for simplicity and robustness.
        return None

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
    ]
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Try multiple starting points to avoid local minima
    best_h = h0
    best_score = _score_raw(h0)
    
    # 1. Uniform
    res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
    if res.success:
        h_res = np.clip(res.x, 0, None)
        h_res /= np.sum(h_res) # Normalize again
        s = _score_raw(h_res)
        if s > best_score:
            best_score = s
            best_h = h_res

    # 2. Sparse random starts
    np.random.seed(42)
    for _ in range(5):
        h_start = np.zeros(N)
        # Pick 10-20 random indices
        k = np.random.randint(10, 20)
        idx = np.random.choice(N, k, replace=False)
        h_start[idx] = 1.0 / k
        res = minimize(objective, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
        if res.success:
            h_res = np.clip(res.x, 0, None)
            h_res /= np.sum(h_res)
            s = _score_raw(h_res)
            if s > best_score:
                best_score = s
                best_h = h_res

    # 3. Sidon set initialization (if we want to combine with previous logic)
    # But the directive says "design a new program... do not reuse current designs".
    # So I will stick to the optimization approach.
    
    return list(best_h)
