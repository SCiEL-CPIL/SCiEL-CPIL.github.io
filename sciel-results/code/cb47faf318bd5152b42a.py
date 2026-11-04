import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 100  # 20s margin
    
    # Mapping (i, j, k) to equation index
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # eq_idx = (i * m + j) * p + k ? No, W index is k*n+i.
    # The equation is for each i, j, k.
    # Let's map i, j, k to a flat index 0..71
    def get_eq_idx(i, j, k):
        return (i * m + j) * p + k
        
    # Pre-compute indices for fast update
    # For each U index a (i*m+j), which equations does it affect?
    # a -> i, j. Affected k: 0..p-1.
    # Eq idx: (i*m+j)*p + k
    
    # For each V index b (j*p+k), which equations?
    # b -> j, k. Affected i: 0..n-1.
    # Eq idx: (i*m+j)*p + k
    
    # For each W index c (k*n+i), which equations?
    # c -> k, i. Affected j: 0..m-1.
    # Eq idx: (i*m+j)*p + k

    def compute_error(U, V, W, r):
        err = [0.0] * (n * m * p)
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for t in range(r):
                        s += U[a][t] * V[b][t] * W[c][t]
                    eq_idx = (i * m + j) * p + k
                    err[eq_idx] = s - 1.0
        return err

    def search_rank(r, time_limit):
        # Try to find a solution of rank r
        # Values to try
        vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
        
        best_err = float('inf')
        
        max_restarts = 1000
        for restart in range(max_restarts):
            if time.time() > time_limit:
                return None, best_err
            
            # Initialize
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            # Random init with some sparsity
            for a in range(n * m):
                for t in range(r):
                    if np.random.random() < 0.3:
                        U[a][t] = vals[np.random.randint(len(vals))]
            for b in range(m * p):
                for t in range(r):
                    if np.random.random() < 0.3:
                        V[b][t] = vals[np.random.randint(len(vals))]
            for c in range(p * n):
                for t in range(r):
                    if np.random.random() < 0.3:
                        W[c][t] = vals[np.random.randint(len(vals))]
            
            err = compute_error(U, V, W, r)
            curr_err = sum(e*e for e in err)
            
            if curr_err < best_err:
                best_err = curr_err
            
            if curr_err == 0:
                return (U, V, W), 0.0

            # Local Search
            no_improve = 0
            max_iters = 5000
            for _ in range(max_iters):
                if time.time() > time_limit:
                    break
                
                # Pick random variable
                var_type = np.random.randint(3) # 0:U, 1:V, 2:W
                if var_type == 0:
                    a = np.random.randint(n * m)
                    t = np.random.randint(r)
                    old_val = U[a][t]
                elif var_type == 1:
                    b = np.random.randint(m * p)
                    t = np.random.randint(r)
                    old_val = V[b][t]
                else:
                    c = np.random.randint(p * n)
                    t = np.random.randint(r)
                    old_val = W[c][t]
                
                # Pick new value
                new_val = vals[np.random.randint(len(vals))]
                if new_val == old_val:
                    continue
                
                delta = new_val - old_val
                
                # Calculate change in error
                # We need to update the error vector
                # This is expensive if we recompute from scratch.
                # But n,m,p are small.
                # Let's just recompute the affected errors.
                
                new_err_sum = 0.0
                old_err_sum = 0.0
                
                # To speed up, we can just recompute the whole error if it's fast enough.
                # 72 equations, r up to 70. 72*70 = 5040 ops.
                # Very fast.
                
                # Save old state? No, just recompute.
                # But we need to compare.
                
                # Apply change
                if var_type == 0:
                    U[a][t] = new_val
                elif var_type == 1:
                    V[b][t] = new_val
                else:
                    W[c][t] = new_val
                
                new_err = compute_error(U, V, W, r)
                new_err_sum = sum(e*e for e in new_err)
                
                if new_err_sum < curr_err:
                    curr_err = new_err_sum
                    err = new_err
                    no_improve = 0
                    if curr_err == 0:
                        return (U, V, W), 0.0
                else:
                    # Revert
                    if var_type == 0:
                        U[a][t] = old_val
                    elif var_type == 1:
                        V[b][t] = old_val
                    else:
                        W[c][t] = old_val
                    no_improve += 1
                    if no_improve > 100:
                        break
                        
        return None, best_err

    # Try ranks
    # Start with lower ranks
    for r in [50, 55, 60, 65, 70, 71]:
        if time.time() > deadline:
            break
        sol, _ = search_rank(r, deadline)
        if sol:
            U, V, W = sol
            return {"U": U, "V": V, "W": W}
            
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
