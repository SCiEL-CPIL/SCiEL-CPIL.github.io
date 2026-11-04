import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initial Solution: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    # Helper to calculate delta cost for swapping indices i and j in permutation p
    def delta_cost(i, j, p):
        li = p[i]
        lj = p[j]
        
        # Direct interaction
        delta = a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        # Interactions with all other k
        delta_sum = 0
        # Use local references for speed
        a_i = a[i]
        a_j = a[j]
        a_col_i = a[:, i]
        a_col_j = a[:, j]
        b_li = b[li]
        b_lj = b[lj]
        b_col_li = b[:, li]
        b_col_lj = b[:, lj]
        p_arr = p
        
        for k in range(n):
            if k == i or k == j:
                continue
            lk = p_arr[k]
            old = a_i[k] * b_li[lk] + a_col_i[k] * b_col_li[lk] + a_j[k] * b_lj[lk] + a_col_j[k] * b_col_lj[lk]
            new = a_i[k] * b_lj[lk] + a_col_i[k] * b_col_lj[lk] + a_j[k] * b_li[lk] + a_col_j[k] * b_col_li[lk]
            delta_sum += new - old
            
        return delta + delta_sum

    # Main ILS Loop
    perturb_size = 5
    
    while time.time() - start_time < time_limit:
        # Local Search Phase (Steepest Descent)
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Find the best improvement in the entire neighborhood
            for i in range(n):
                for j in range(i + 1, n):
                    dc = delta_cost(i, j, p)
                    if dc < best_delta:
                        best_delta = dc
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                # Apply the best move
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        # Perturbation Phase
        # Swap 'perturb_size' random pairs to escape local optimum
        for _ in range(perturb_size):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Recalculate cost after perturbation
        current_cost = calc_cost(p)
        
        # Check time
        if time.time() - start_time > time_limit:
            break

    return best_p.tolist()
