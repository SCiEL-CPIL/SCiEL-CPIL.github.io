import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Initialization: Rank Matching
    # Heavy flows to short distances
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by flow (descending) and locations by distance (ascending)
    heavy_order = np.argsort(-a_sums, kind="stable")
    central_order = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # 2. Efficient Cost Calculation and Delta Update
    # We maintain the permuted distance matrix B_p where B_p[i, j] = B[p[i], p[j]]
    # Cost = sum(a * B_p)
    # To speed up, we precompute row and column sums of a? No, we need element-wise product.
    
    # Let's use a vectorized approach for the cost calculation and delta.
    # B_p = b[np.ix_(p, p)]
    # cost = np.sum(a * B_p)
    
    # For a swap of i and j:
    # Let x = p[i], y = p[j]
    # New p: p'[i]=y, p'[j]=x, others same.
    # B_p changes only in row i, col i, row j, col j.
    # Specifically:
    # B_p'[i, :] = B[y, p']  -> B[y, p] except at j where it is B[y, x]
    # B_p'[:, i] = B[p', y]  -> B[p, y] except at j where it is B[x, y]
    # B_p'[j, :] = B[x, p']  -> B[x, p] except at i where it is B[x, y]
    # B_p'[:, j] = B[p', x]  -> B[p, x] except at i where it is B[y, x]
    
    # Let's compute the delta cost analytically.
    # Old cost terms involving i or j:
    # C_old_part = sum_{k} a[i,k] B[x, p[k]] + sum_{k} a[k,i] B[p[k], x] 
    #             + sum_{k} a[j,k] B[y, p[k]] + sum_{k} a[k,j] B[p[k], y]
    #             - a[i,i] B[x,x] - a[j,j] B[y,y]  (diagonals counted twice in sums above? 
    #             Wait, sum_{k} a[i,k] B[x, p[k]] includes k=i -> a[i,i]B[x,x].
    #             sum_{k} a[k,i] B[p[k], x] includes k=i -> a[i,i]B[x,x].
    #             So a[i,i]B[x,x] is counted twice. Same for j.
    #             Also terms a[i,j]B[x,y] and a[j,i]B[y,x] are in the sums.
    
    # Let's just compute the change directly using vectors.
    # Let A_i = a[i, :] (row i of A)
    # Let A_j = a[j, :] (row j of A)
    # Let A^T_i = a[:, i] (col i of A)
    # Let A^T_j = a[:, j] (col j of A)
    
    # Old contributions from row i: A_i @ B[x, p]  (where B[x, p] is the vector B[x, p[0]], B[x, p[1]], ...])
    # Note: B[x, p] is just the row x of B, indexed by p.
    # Let B_x_p = b[x, p]
    # Let B_y_p = b[y, p]
    # Let B_p_x = b[p, x]
    # Let B_p_y = b[p, y]
    
    # Old Row i cost: a[i, :] @ b[x, p]
    # Old Col i cost: a[:, i] @ b[p, x]
    # Old Row j cost: a[j, :] @ b[y, p]
    # Old Col j cost: a[:, j] @ b[p, y]
    
    # New p: p' differs at i, j.
    # p'[i] = y, p'[j] = x.
    # New Row i cost: a[i, :] @ b[y, p']
    # Note b[y, p'] is b[y, p] but with element j changed from b[y, x] to b[y, y] ? No.
    # p'[j] = x, so b[y, p'[j]] = b[y, x]. Wait, p[j] was y, so b[y, p[j]] = b[y, y].
    # So b[y, p'] is b[y, p] with index i changed? No, p' is the permutation.
    # The vector v = b[y, p] has v[k] = b[y, p[k]].
    # v'[k] = b[y, p'[k]].
    # v'[i] = b[y, p'[i]] = b[y, y].
    # v'[j] = b[y, p'[j]] = b[y, x].
    # v'[k] = v[k] for k != i, j.
    
    # So New Row i cost = a[i, :] @ v' = a[i, :] @ v + a[i, i](b[y, y] - b[y, y]) + a[i, j](b[y, x] - b[y, y])
    # Wait, v[i] was b[y, p[i]] = b[y, x]. v'[i] = b[y, y].
    # v[j] was b[y, p[j]] = b[y, y]. v'[j] = b[y, x].
    # So change in Row i cost = a[i, i](b[y, y] - b[y, x]) + a[i, j](b[y, x] - b[y, y])
    #                          = (a[i, i] - a[i, j]) * (b[y, y] - b[y, x])
    
    # Similarly for others? This seems prone to error.
    
    # Alternative: Full re-evaluation is O(n^2). With n=150, it's ~22,500 ops.
    # In NumPy, this is extremely fast.
    # Let's benchmark: np.sum(a * b[np.ix_(p, p)])
    # If this takes ~10-20 microseconds, we can do 50,000-100,000 iterations in 1 second.
    # In 140 seconds, we can do 7-14 million iterations.
    # This is a massive search space coverage compared to complex delta updates which might be slower in Python overhead.
    
    # Let's try the simple full re-evaluation first. It's robust and likely fast enough in NumPy.
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_p = p.copy()
    current_cost = compute_cost(current_p)
    best_p = current_p.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    # T0 = 1000.0 (arbitrary, scale with cost magnitude?)
    # Cost is ~8e6. Let's start T0 = 1000.
    T = 1000.0
    T_min = 0.01
    cooling_rate = 0.9999  # Slow cooling
    
    # To allow large jumps, we can use "Or-opt" moves (move a block) or just random swaps.
    # The prompt suggests "digit-reversal" or non-adjacent swaps. Random swaps are non-adjacent.
    # Let's stick to random swaps for simplicity and speed.
    
    while time.time() < deadline:
        # Pick two distinct indices
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        # Swap
        current_p[i], current_p[j] = current_p[j], current_p[i]
        
        new_cost = compute_cost(current_p)
        
        delta = new_cost - current_cost
        
        if delta < 0:
            # Accept improvement
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = current_p.copy()
        else:
            # Accept worse move with probability exp(-delta / T)
            if delta < T and random.random() < np.exp(-delta / T):
                current_cost = new_cost
            else:
                # Revert
                current_p[i], current_p[j] = current_p[j], current_p[i]
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Optional: If stuck for a long time, restart with perturbation?
        # SA should handle this.
        
    return best_p.tolist()
