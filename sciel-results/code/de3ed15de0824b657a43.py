"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Ensure matrices are int64 for consistent arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    # --- 1. Initial Solution: Rank-Matching ---
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    p_list = p.tolist()
    
    # --- 2. Local Search with O(n^2) Swap Evaluation ---
    
    def compute_cost(p_list):
        p_arr = np.array(p_list)
        total = 0
        for i in range(n):
            pi = p_arr[i]
            total += np.dot(a[i, :], b[pi, p_arr])
        return total

    current_cost = compute_cost(p_list)
    best_p = p_list[:]
    best_cost = current_cost
    
    p_np = np.array(p_list)
    b_diag = np.diag(b)
    a_diag = np.diag(a)
    
    def find_best_swap():
        best_delta = 0
        best_i = -1
        best_k = -1
        
        for i in range(n):
            u = p_np[i]
            a_i = a[i, :]
            a_ii = a_diag[i]
            b_u = b[u, :]
            b_uu = b_diag[u]
            
            for k in range(i + 1, n):
                v = p_np[k]
                a_k = a[k, :]
                a_kk = a_diag[k]
                b_v = b[v, :]
                b_vv = b_diag[v]
                
                # Term 1: Diagonal
                term1 = (a_ii - a_kk) * (b_vv - b_uu)
                
                # Term 2: Interaction
                d_a = a_i - a_k
                d_b = b_v - b_u
                d_b_perm = d_b[p_np]
                
                total_sum = np.dot(d_a, d_b_perm)
                
                sub_i = d_a[i] * d_b[u]
                sub_k = d_a[k] * d_b[v]
                
                interaction_sum = total_sum - sub_i - sub_k
                
                delta = term1 + 2 * interaction_sum
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_k = k
        
        return best_delta, best_i, best_k
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        best_delta, best_i, best_k = find_best_swap()
        
        if best_i != -1:
            # Perform swap
            p_list[best_i], p_list[best_k] = p_list[best_k], p_list[best_i]
            p_np[best_i], p_np[best_k] = p_np[best_k], p_np[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p_list[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
    return best_p
# EVOLVE-BLOCK-END
