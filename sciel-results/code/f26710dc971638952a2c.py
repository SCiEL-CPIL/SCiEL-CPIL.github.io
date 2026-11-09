import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Initial guess: uniform
    P_init = np.ones((n, n)) / (n * n)
    
    def calc_entropies(P):
        # P is 10x10
        # X
        px = P.sum(axis=0)
        hx = -np.sum(px[px > 0] * np.log2(px[px > 0]))
        
        # X - Y
        # values x - y
        # y in 0..9, x in 0..9
        # k = x - y
        # prob of k is sum P[y, x] where x-y=k
        # x = y + k
        # y from max(0, -k) to min(9, 9-k)
        p_d = np.zeros(19) # k from -9 to 9
        for y in range(n):
            for x in range(n):
                k = x - y + 9
                p_d[k] += P[y, x]
        hd = -np.sum(p_d[p_d > 0] * np.log2(p_d[p_d > 0]))
        
        # X + Y
        # s = x + y
        # s from 0 to 18
        p_s = np.zeros(19)
        for y in range(n):
            for x in range(n):
                s = x + y
                p_s[s] += P[y, x]
        hs = -np.sum(p_s[p_s > 0] * np.log2(p_s[p_s > 0]))
        
        # X + 2Y
        # t = x + 2y
        # t from 0 to 27
        p_t = np.zeros(28)
        for y in range(n):
            for x in range(n):
                t = x + 2 * y
                p_t[t] += P[y, x]
        ht = -np.sum(p_t[p_t > 0] * np.log2(p_t[p_t > 0]))
        
        return hx, hd, hs, ht

    def objective(params):
        P = params.reshape((n, n))
        hx, hd, hs, ht = calc_entropies(P)
        m = max(hx, hs, ht)
        # We want to maximize hd / m
        # Equivalent to minimize m / hd or -hd + m
        # Let's try maximizing hd - m
        if m == 0:
            return 0
        return -(hd - m)
        
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda P: np.sum(P) - 1}
    ]
    
    # Bounds
    bounds = [(0, 1) for _ in range(n * n)]
    
    # Optimization
    # SLSQP might be slow or get stuck.
    # Let's try a few random restarts or a specific heuristic.
    
    # Heuristic: Start with column 0 (score 1.0) and try to perturb.
    # Or start with uniform.
    
    best_P = P_init
    best_score = -np.inf
    
    # Try optimizing from uniform
    res = minimize(objective, P_init.flatten(), method='SLSQP', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-6})
    if res.success:
        P_opt = res.x.reshape((n, n))
        hx, hd, hs, ht = calc_entropies(P_opt)
        m = max(hx, hs, ht)
        if m > 1e-9:
            score = hd / m
            if score > best_score:
                best_score = score
                best_P = P_opt
                
    # Try optimizing from column 0
    P_col = np.zeros((n, n))
    P_col[:, 0] = 1.0 / n
    res2 = minimize(objective, P_col.flatten(), method='SLSQP', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-6})
    if res2.success:
        P_opt2 = res2.x.reshape((n, n))
        hx, hd, hs, ht = calc_entropies(P_opt2)
        m = max(hx, hs, ht)
        if m > 1e-9:
            score = hd / m
            if score > best_score:
                best_score = score
                best_P = P_opt2

    # Normalize and clip
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total > 1e-9:
        best_P /= total
    else:
        best_P = P_init # Fallback
        
    return best_P
