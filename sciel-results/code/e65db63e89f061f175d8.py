import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Initial permutation
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute d_ij? 
    # d_ij = a[i] - a[j]
    # We can compute on the fly or precompute a list of d_ij vectors.
    # Precomputing d_ij for all i<j takes O(n^2) memory, which is fine (5000 * 100 ints).
    
    d_mats = np.zeros((n, n, n), dtype=np.int64) # Too much memory? 100*100*100 = 1M ints = 8MB. Fine.
    # Actually, just compute d_i_j = a[i] - a[j] on the fly is faster than memory access?
    # a[i] is a view. a[i] - a[j] creates a new array. 
    # Precomputation:
    # d_all = a[:, np.newaxis, :] - a[np.newaxis, :, :] # Shape (n, n, n)
    # d_all[i, j] is a[i] - a[j].
    
    d_all = a[:, np.newaxis, :] - a[np.newaxis, :, :]
    
    # v_xy = b[x] - b[y]
    # v_all = b[:, np.newaxis, :] - b[np.newaxis, :, :] # Shape (n, n, n)
    # v_all[x, y] is b[x] - b[y].
    
    v_all = b[:, np.newaxis, :] - b[np.newaxis, :, :]
    
    def get_cost(p):
        # sum a[i,j] b[p[i], p[j]]
        # Efficiently: 
        # B_p = b[np.ix_(p, p)]
        # return np.sum(a * B_p)
        return np.sum(a * b[np.ix_(p, p)])
        
    current_cost = get_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = start_time + 110 # 110s limit to be safe
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Loop over all pairs
        for i in range(n):
            for j in range(i + 1, n):
                x = p[i]
                y = p[j]
                
                # d_ij = d_all[i, j]
                # v_xy = v_all[x, y]
                # v_xy_p = v_xy[p]
                
                # Delta = dot(d_ij, v_xy_p) - d_ij[i]*v_xy[x] - d_ij[j]*v_xy[y]
                
                d_ij = d_all[i, j]
                v_xy = v_all[x, y]
                v_xy_p = v_xy[p]
                
                delta = np.dot(d_ij, v_xy_p)
                # Subtract terms for i and j
                delta -= d_ij[i] * v_xy[x]
                delta -= d_ij[j] * v_xy[y]
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            # Perform swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
            # Optional: break early if we hit time limit? 
            # The loop checks at the top.
            
    return best_p.tolist()
