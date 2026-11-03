"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    # 1. Initialization: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    central_order = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    def calc_cost(p):
        b_perm = b[:, p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Tabu parameters
    tabu_tenure = 10
    tabu_expiry = np.zeros((n, n), dtype=np.int64)  # tabu_expiry[i,j] = iteration when (i,j) is no longer tabu
    iteration = 0
    no_improve_count = 0
    max_no_improve = 50
    
    is_symmetric_a = np.allclose(a, a.T)
    is_symmetric_b = np.allclose(b, b.T)
    
    while time.time() < deadline:
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        B_p = b[:, p]       # B_p[x, k] = b[x, p[k]]
        B_pT = b[p, :]      # B_pT[k, x] = b[p[k], x]
        
        for i in range(n):
            u = p[i]
            
            if is_symmetric_a and is_symmetric_b:
                diff_b_all = B_p[p] - B_p[u]
                diff_a_all = a[i][np.newaxis, :] - a
                main_dots = np.sum(diff_a_all * diff_b_all, axis=1)
                
                b_vu = B_pT[:, u]
                b_uu = b[u, u]
                a_ii = a[i, i]
                a_ji = a[:, i]
                a_ij = a[i, :]
                a_jj = np.diag(a)
                b_vv = np.diag(b[p])
                b_uv = B_p[u, :]
                
                term_i = (a_ii - a_ji) * (b_vu - b_uu)
                term_j = (a_ij - a_jj) * (b_vv - b_uv)
                
                deltas = 2 * (main_dots - term_i - term_j)
            else:
                diff_b_all = B_p[p] - B_p[u]
                diff_a_all = a[i][np.newaxis, :] - a
                sum1_main = np.sum(diff_a_all * diff_b_all, axis=1)
                
                b_vu = B_pT[:, u]
                b_uu = b[u, u]
                a_ii = a[i, i]
                a_ji = a[:, i]
                a_ij = a[i, :]
                a_jj = np.diag(a)
                b_vv = np.diag(b[p])
                b_uv = B_p[u, :]
                
                term1_i = (a_ii - a_ji) * (b_vu - b_uu)
                term1_j = (a_ij - a_jj) * (b_vv - b_uv)
                sum1 = sum1_main - term1_i - term1_j
                
                diff_a_col_all = a[:, i][np.newaxis, :] - a.T
                diff_b_col_all = B_pT[:, p] - B_pT[:, u][np.newaxis, :]
                sum2_main = np.sum(diff_a_col_all * diff_b_col_all, axis=0)
                
                b_uv2 = B_p[u, :]
                term2_i = (a_ii - a_ij) * (b_uv2 - b_uu)
                b_vj_vj_u = b_vv - b_vu
                term2_j = (a_ji - a_jj) * b_vj_vj_u
                sum2 = sum2_main - term2_i - term2_j
                
                b_uvj = B_p[u, :]
                direct = (a_ij * (b_vu - b_uvj) +
                         a_ji * (b_uvj - b_vu) +
                         a_ii * (b_vv - b_uu) +
                         a_jj * (b_uu - b_vv))
                
                deltas = sum1 + sum2 + direct
            
            deltas[i] = np.inf
            
            # Apply tabu: mask out tabu moves that don't satisfy aspiration
            tabu_mask = tabu_expiry[i, :] > iteration
            if np.any(tabu_mask):
                # For tabu moves, only allow if new cost < best_cost (aspiration)
                tabu_j = np.where(tabu_mask)[0]
                for j in tabu_j:
                    if current_cost + deltas[j] >= best_cost:
                        deltas[j] = np.inf
            
            j_best = np.argmin(deltas)
            if deltas[j_best] < best_delta:
                best_delta = deltas[j_best]
                best_i = i
                best_j = j_best
        
        if best_i == -1:
            # All moves are tabu and no aspiration; pick random
            for _ in range(100):
                best_i = random.randint(0, n - 1)
                best_j = random.randint(0, n - 1)
                if best_i != best_j:
                    break
            u = p[best_i]
            v = p[best_j]
            if is_symmetric_a and is_symmetric_b:
                diff_a = a[best_i] - a[best_j]
                diff_b = B_p[v] - B_p[u]
                full_dot = np.dot(diff_a, diff_b)
                term_i = (a[best_i, best_i] - a[best_j, best_i]) * (b[v, u] - b[u, u])
                term_j = (a[best_i, best_j] - a[best_j, best_j]) * (b[v, v] - b[u, v])
                best_delta = 2 * (full_dot - term_i - term_j)
            else:
                diff_a_row = a[best_i] - a[best_j]
                diff_b_row = B_p[v] - B_p[u]
                sum1_full = np.dot(diff_a_row, diff_b_row)
                term1_i = (a[best_i, best_i] - a[best_j, best_i]) * (b[v, u] - b[u, u])
                term1_j = (a[best_i, best_j] - a[best_j, best_j]) * (b[v, v] - b[u, v])
                sum1 = sum1_full - term1_i - term1_j
                diff_a_col = a[:, best_i] - a[:, best_j]
                diff_b_col = B_pT[:, v] - B_pT[:, u]
                sum2_full = np.dot(diff_a_col, diff_b_col)
                term2_i = (a[best_i, best_i] - a[best_i, best_j]) * (b[u, v] - b[u, u])
                term2_j = (a[best_j, best_i] - a[best_j, best_j]) * (b[v, v] - b[v, u])
                sum2 = sum2_full - term2_i - term2_j
                direct = (a[best_i, best_j] * (b[v, u] - b[u, v]) +
                          a[best_j, best_i] * (b[u, v] - b[v, u]) +
                          a[best_i, best_i] * (b[v, v] - b[u, u]) +
                          a[best_j, best_j] * (b[u, u] - b[v, v]))
                best_delta = sum1 + sum2 + direct
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)
        
        # Update Tabu
        tabu_expiry[best_i, best_j] = iteration + tabu_tenure
        tabu_expiry[best_j, best_i] = iteration + tabu_tenure
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        iteration += 1
        
        if no_improve_count > max_no_improve:
            for _ in range(5):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
            tabu_expiry[:] = 0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
