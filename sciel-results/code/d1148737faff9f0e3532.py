import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Helper to calculate cost
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    # Helper to do local search (First Improvement 2-opt)
    def local_search(p):
        p = p.copy()
        current_cost = calc_cost(p)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, current_cost
                    
                    x = p[i]
                    y = p[j]
                    
                    # Calculate delta cost
                    # Terms:
                    # 1. Direct i-j interaction change
                    direct_delta = a[i, j] * (b[y, x] - b[x, y]) + a[j, i] * (b[x, y] - b[y, x])
                    
                    # 2. Interaction with other k
                    # Sum over k != i, j of (a[i,k] - a[j,k]) * (b[y, p[k]] - b[x, p[k]]) 
                    # + (a[k,i] - a[k,j]) * (b[p[k], y] - b[p[k], x])
                    
                    # Vectorized computation
                    # diff_row_vec = a[i, :] - a[j, :]
                    # diff_col_vec = a[:, i] - a[:, j]
                    
                    # We need b[y, p] - b[x, p] and b[p, y] - b[p, x]
                    # b[y, p] is b[y][p]
                    # b[x, p] is b[x][p]
                    # b[p, y] is b[:, y][p]
                    # b[p, x] is b[:, x][p]
                    
                    # To avoid O(n) overhead for each pair, we can precompute?
                    # No, p changes.
                    
                    # This is O(n) per pair.
                    # n=100, so 100 ops.
                    # 5000 pairs * 100 ops = 500k ops per sweep.
                    # This is fine.
                    
                    diff_row = a[i, :] - a[j, :]
                    diff_col = a[:, i] - a[:, j]
                    
                    # b[y, p] - b[x, p]
                    term1 = np.dot(diff_row, b[y, p] - b[x, p])
                    
                    # b[p, y] - b[p, x]
                    term2 = np.dot(diff_col, b[:, y][p] - b[:, x][p])
                    
                    delta = direct_delta + term1 + term2
                    
                    if delta < -1e-9: # Strictly better
                        # Perform swap
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        # Restart inner loop? Or continue?
                        # First improvement usually continues, but restarting can be more effective.
                        # Let's continue for speed, but break if we want to re-evaluate from start.
                        # Standard first improvement: continue scanning.
                        pass
        
        return p, current_cost

    best_p = None
    best_cost = float('inf')
    
    # 1. Rank Matching Start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    p_res, cost_res = local_search(p_init)
    if cost_res < best_cost:
        best_cost = cost_res
        best_p = p_res.copy()
        
    if time.time() > deadline:
        return best_p.tolist()

    # 2. Random Restarts
    # How many random restarts can we afford?
    # Each local search sweep is ~0.1s? Let's estimate.
    # If local search takes 0.5s per run, we can do ~200 runs.
    # Let's try 50 random restarts.
    
    np.random.seed(42) # Deterministic for reproducibility if needed, but random is fine
    
    for k in range(50):
        if time.time() > deadline:
            break
            
        p_rand = np.random.permutation(n)
        p_res, cost_res = local_search(p_rand)
        if cost_res < best_cost:
            best_cost = cost_res
            best_p = p_res.copy()
            
    return best_p.tolist()
