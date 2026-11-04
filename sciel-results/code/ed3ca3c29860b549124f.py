"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 118.0
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    def delta_swap(p, i, j):
        pi = p[i]
        pj = p[j]
        
        # Optimized delta calculation
        old_row_i = np.dot(a[i, :], b[pi, p])
        old_col_i = np.dot(a[:, i], b[p, pi])
        old_row_j = np.dot(a[j, :], b[pj, p])
        old_col_j = np.dot(a[:, j], b[p, pj])
        
        old_diag_corr = a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj]
        old_affected = old_row_i + old_col_i + old_row_j + old_col_j - old_diag_corr
        
        base_row_i = np.dot(a[i, :], b[pj, p])
        new_row_i = base_row_i - a[i,i]*b[pj,pi] - a[i,j]*b[pj,pj] + a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi]
        
        base_col_i = np.dot(a[:, i], b[p, pj])
        new_col_i = base_col_i - a[i,i]*b[pi,pj] - a[j,i]*b[pj,pj] + a[i,i]*b[pj,pj] + a[j,i]*b[pi,pj]
        
        base_row_j = np.dot(a[j, :], b[pi, p])
        new_row_j = base_row_j - a[j,i]*b[pi,pi] - a[j,j]*b[pi,pj] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
        
        base_col_j = np.dot(a[:, j], b[p, pi])
        new_col_j = base_col_j - a[i,j]*b[pi,pi] - a[j,j]*b[pj,pi] + a[i,j]*b[pj,pi] + a[j,j]*b[pi,pi]
        
        new_diag_corr = a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
        
        new_affected = new_row_i + new_col_i + new_row_j + new_col_j - new_diag_corr
        
        return new_affected - old_affected

    # NEW INITIALIZATION: Rank Sort
    # 1. Compute row sums for flow and distance
    a_rs = a.sum(axis=1)
    b_rs = b.sum(axis=1)
    
    # Strategy 1: High Flow -> Low Distance
    fac_order_1 = np.argsort(-a_rs) # Descending
    loc_order_1 = np.argsort(b_rs)  # Ascending
    p1 = np.empty(n, dtype=np.int64)
    p1[fac_order_1] = loc_order_1
    cost1 = calc_cost(p1)
    
    # Strategy 2: High Flow -> High Distance (Reverse)
    fac_order_2 = np.argsort(-a_rs)
    loc_order_2 = np.argsort(-b_rs)
    p2 = np.empty(n, dtype=np.int64)
    p2[fac_order_2] = loc_order_2
    cost2 = calc_cost(p2)
    
    # Strategy 3: Low Flow -> Low Distance
    fac_order_3 = np.argsort(a_rs)
    loc_order_3 = np.argsort(b_rs)
    p3 = np.empty(n, dtype=np.int64)
    p3[fac_order_3] = loc_order_3
    cost3 = calc_cost(p3)
    
    # Strategy 4: Random
    p4 = np.random.permutation(n)
    cost4 = calc_cost(p4)
    
    # Pick the best initial solution
    candidates = [(cost1, p1), (cost2, p2), (cost3, p3), (cost4, p4)]
    candidates.sort(key=lambda x: x[0])
    best_cost = candidates[0][0]
    p = candidates[0][1].copy()
    best_p = p.copy()
    
    # Simulated Annealing
    T_start = 500.0
    T_end = 1.0
    cooling_rate = 0.9999
    
    while time.time() - start_time < time_limit:
        T = T_start
        no_improve_counter = 0
        max_no_improve = 5000
        
        while T > T_end and time.time() - start_time < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            delta = delta_swap(p, i, j)
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                best_cost_current = best_cost # shadow variable not needed, just update best if improved
                if delta < 0:
                    # It's an improvement
                    pass # cost update implicit in delta logic if we track current_cost
                # We need to track current_cost properly
                # Let's restart the SA loop structure to be cleaner
                break 
        # The above break logic is messy. Let's stick to the previous structure but with new init.
        
    # Let's just use the previous SA structure which was working, just with new init.
    
    # Re-implementing SA loop cleanly
    current_cost = best_cost
    T_start = 1000.0
    T_end = 1.0
    cooling_rate = 0.99995
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        T = T_start
        no_improve_counter = 0
        
        while T > T_end and time.time() - start_time < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            delta = delta_swap(p, i, j)
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            
            T *= cooling_rate
            
            if no_improve_counter > max_no_improve:
                break
        
        if no_improve_counter > max_no_improve and time.time() - start_time < time_limit:
            p = best_p.copy()
            current_cost = best_cost
    
    # Final Local Search Polish
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = delta_swap(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
