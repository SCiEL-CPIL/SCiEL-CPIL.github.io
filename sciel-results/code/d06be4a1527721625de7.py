# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115.0
    
    # 1. Initial Assignment: Greedy Rank Matching
    # Sort facilities by total flow (descending)
    # Sort locations by total distance (ascending)
    # This is a strong heuristic starting point.
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    fac_order = np.argsort(-a_sum, kind='stable')
    loc_order = np.argsort(b_sum, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    def compute_cost(perm):
        return np.sum(a * b[np.ix_(perm, perm)])
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for faster delta
    # We need to calculate delta for swapping i and j.
    # Let li = p[i], lj = p[j].
    # Delta = sum_{k != i,j} a[i,k]*(b[lj,k]-b[li,k]) + a[j,k]*(b[li,k]-b[lj,k]) 
    #         + sum_{k != i,j} a[k,i]*(b[k,lj]-b[k,li]) + a[k,j]*(b[k,li]-b[k,lj])
    #         + (a[i,j]+a[j,i])*(b[li,lj]-b[lj,li])
    
    # To speed up, we can precompute row differences? No, li/lj change.
    # We will use numpy dot products.
    
    # Simulated Annealing Parameters
    T = 1000.0 # Initial temperature, scale with problem size/cost
    T_min = 1e-4
    alpha = 0.995 # Cooling rate
    
    # Estimate cost scale
    max_flow = np.max(a)
    max_dist = np.max(b)
    T = 0.1 * n * max_flow * max_dist / 100.0 # Heuristic init
    
    # To avoid division by zero or issues, let's normalize temperature
    # A better T init is based on average edge cost change
    # Let's just use a robust T
    T = 500.0
    
    while time.time() - start_time < time_budget - 2.0:
        if T < T_min:
            break
            
        # Pick random pair i, j
        i, j = random.sample(range(n), 2)
        
        li = p[i]
        lj = p[j]
        
        # Calculate Delta
        # Vectorized calculation
        # diff_b_row = b[lj, :] - b[li, :]
        # diff_b_col = b[:, lj] - b[:, li]
        
        # Sum 1: Contribution of row i and j to other columns
        # sum_{k != i,j} a[i,k] * (b[lj,k] - b[li,k])
        # sum_{k != i,j} a[j,k] * (b[li,k] - b[lj,k])
        
        # It is faster to compute full sums and subtract i,j terms
        diff_row = b[lj, :] - b[li, :]
        diff_col = b[:, lj] - b[:, li]
        
        # Term 1: a[i, :] dot diff_row
        s1 = np.dot(a[i, :], diff_row)
        # Subtract terms where k=i or k=j
        # k=i: a[i,i] * (b[lj,i] - b[li,i])
        # k=j: a[i,j] * (b[lj,j] - b[li,j])
        s1 -= a[i, i] * (b[lj, i] - b[li, i])
        s1 -= a[i, j] * (b[lj, j] - b[li, j])
        
        # Term 2: a[j, :] dot (-diff_row)  => -a[j, :] dot diff_row
        s2 = -np.dot(a[j, :], diff_row)
        # Subtract terms where k=i or k=j
        # k=i: -a[j,i] * (b[lj,i] - b[li,i])
        # k=j: -a[j,j] * (b[lj,j] - b[li,j])
        s2 -= a[j, i] * (b[lj, i] - b[li, i])
        s2 -= a[j, j] * (b[lj, j] - b[li, j])
        
        # Term 3: a[:, i] dot diff_col
        s3 = np.dot(a[:, i], diff_col)
        # Subtract terms where k=i or k=j
        # k=i: a[i,i] * (b[i,lj] - b[i,li])
        # k=j: a[j,i] * (b[j,lj] - b[j,li])
        s3 -= a[i, i] * (b[i, lj] - b[i, li])
        s3 -= a[j, i] * (b[j, lj] - b[j, li])
        
        # Term 4: a[:, j] dot (-diff_col) => -a[:, j] dot diff_col
        s4 = -np.dot(a[:, j], diff_col)
        # Subtract terms where k=i or k=j
        # k=i: -a[i,j] * (b[i,lj] - b[i,li])
        # k=j: -a[j,j] * (b[j,lj] - b[j,li])
        s4 -= a[i, j] * (b[i, lj] - b[i, li])
        s4 -= a[j, j] * (b[j, lj] - b[j, li])
        
        # Term 5: Direct interaction between i and j
        # (a[i,j] + a[j,i]) * (b[li,lj] - b[lj,li])
        s5 = (a[i, j] + a[j, i]) * (b[li, lj] - b[lj, li])
        
        delta = s1 + s2 + s3 + s4 + s5
        
        # Acceptance Criterion
        if delta < 0:
            accept = True
        else:
            if T > 0:
                accept = random.random() < np.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            # Swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= alpha
        
        # Optional: If stuck for a long time, restart or perturb?
        # For now, simple SA.
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
