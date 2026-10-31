import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Initial guess: uniform
    h0 = np.full(N, 0.5)
    
    def overlap_max(h):
        # Compute max overlap
        max_ov = 0
        for k in range(-N, N+1):
            start = max(0, k)
            end = min(N, N+k)
            if start >= end:
                continue
            h1 = h[start:end]
            h2 = h[start-k:end-k]
            ov = np.sum(h1 * (1 - h2)) * (2.0 / N)
            max_ov = max(max_ov, ov)
        return max_ov

    # Constraints
    # sum(h) = N/2
    # 0 <= h <= 1
    
    # We want to minimize overlap_max(h)
    # This function is not differentiable, so we use a smooth approximation or a different strategy.
    # Alternatively, we can maximize the minimum correlation.
    
    # Let's try to maximize min_k C_k
    # C_k = sum(h_i * h_{i-k}) * (2/N)
    
    def min_correlation(h):
        min_c = np.inf
        for k in range(-N, N+1):
            start = max(0, k)
            end = min(N, N+k)
            if start >= end:
                c = 0
            else:
                h1 = h[start:end]
                h2 = h[start-k:end-k]
                c = np.sum(h1 * h2) * (2.0 / N)
            min_c = min(min_c, c)
        return -min_c # Minimize negative min correlation

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2},
        {'type': 'ineq', 'fun': lambda h: h}, # h >= 0
        {'type': 'ineq', 'fun': lambda h: 1 - h} # h <= 1
    ]

    # Use SLSQP
    res = minimize(min_correlation, h0, method='SLSQP', bounds=[(0, 1)]*N, constraints=constraints)
    
    if res.success:
        return res.x
    else:
        return h0
