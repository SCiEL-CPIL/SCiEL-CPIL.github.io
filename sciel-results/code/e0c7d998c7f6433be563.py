"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill
    
    # Precompute for fast cost evaluation
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can rewrite: for a permutation p, cost = sum_i sum_j a[i,j] b[p[i], p[j]]
    # This is equivalent to: sum_i (row i of a dotted with row p[i] of b, but summed over j)
    # More efficiently: cost = trace(a @ B) where B[i,j] = b[p[i], p[j]]
    # Or: cost = sum_i sum_j a[i,j] b[p[i], p[j]]
    #      = sum_i (a[i,:] . b[p[i], :])  -- NO, that's not right either
    # Actually: cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # For a fixed i, the contribution is sum_j a[i,j] b[p[i], p[j]]
    # This is a[i,:] dot the vector [b[p[i],0], b[p[i],1], ..., b[p[i],n-1]]
    # But that depends on p[j] for each j, so it's not a simple dot product per row.
    
    # Better: use the fact that cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # We can compute this as: for each pair (i,j), a[i,j] * b[p[i], p[j]]
    # Vectorized: create matrix C where C[i,j] = a[i,j] * b[p[i], p[j]]
    # Then cost = C.sum()
    # b[p[i], p[j]] for all i,j: this is b[np.ix_(p, p)].sum() * a... no.
    # Actually: b[np.ix_(p, p)] gives a matrix where element (i,j) is b[p[i], p[j]]
    # So cost = (a * b[np.ix_(p, p)]).sum()
    
    def compute_cost(p):
        # p is a numpy array of length n
        return np.sum(a * b[np.ix_(p, p)])
    
    # Initial solution: rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # Simulated Annealing
    T = 30000.0
    T_min = 1.0
    cooling_rate = 0.9995
    
    # Precompute row sums of a for delta computation
    # When we swap p[i] and p[j], the change in cost is:
    # Original contribution involving i or j:
    #   sum_k a[i,k] b[p[i], p[k]] + sum_k a[k,i] b[p[k], p[j]] ... this gets complex.
    # Let's just recompute the full cost for simplicity, but that's O(n^2) per swap.
    # For n=100, n^2 = 10000, which is fast. Let's try it.
    
    no_improve = 0
    restart_threshold = 50000
    
    while time.time() < deadline:
        # Pick two random positions to swap
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
        
        # Swap
        p[i], p[j] = p[j], p[i]
        new_cost = compute_cost(p)
        delta = new_cost - best_cost  # Wait, this is wrong. delta should be new_cost - current_cost
        # Actually, I need to track current cost, not just best.
        # Let me restructure.
        
        # Swap back for now, let me think...
        p[i], p[j] = p[j], p[i]
        break  # Let me restart the logic properly
    
    # Let me rewrite the SA loop properly
    p = best_p.copy()
    current_cost = best_cost
    T = 30000.0
    
    while time.time() < deadline:
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
        
        p[i], p[j] = p[j], p[i]
        new_cost = compute_cost(p)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            p[i], p[j] = p[j], p[i]
            no_improve += 1
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # Adaptive restart
        if no_improve > restart_threshold:
            p = best_p.copy()
            current_cost = best_cost
            T = 30000.0
            no_improve = 0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
