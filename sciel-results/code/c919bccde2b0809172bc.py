# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute constants
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # Check symmetry for gradient computation
    sym = bool(np.allclose(a, a.T, rtol=1e-12, atol=1e-12) and np.allclose(b, b.T, rtol=1e-12, atol=1e-12))

    # --- Helper: compute cost ---
    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    # --- Helper: 2-opt delta (vectorized) ---
    def calc_delta(i, j, perm):
        u = perm[i]
        v = perm[j]
        if u == v:
            return 0

        # Precompute differences
        diff_a_row = a[i, :] - a[j, :]
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[v, :] - b[u, :]
        diff_b_col = b[:, v] - b[:, u]

        # Vectorized computation
        C = diff_b_row[perm]
        D = diff_b_col[perm]

        # Full sums
        full_sum_1 = np.dot(diff_a_row, C)
        full_sum_2 = np.dot(diff_a_col, D)

        # Corrections for diagonal terms (i,i) and (j,j)
        corr_i_1 = diff_a_row[i] * C[i]
        corr_i_2 = diff_a_col[i] * D[i]
        corr_j_1 = diff_a_row[j] * C[j]
        corr_j_2 = diff_a_col[j] * D[j]

        sum_off_diag = (full_sum_1 - corr_i_1 - corr_j_1) + (full_sum_2 - corr_i_2 - corr_j_2)

        # Direct term for (i,j) and (j,i)
        direct_delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])

        return int(sum_off_diag + direct_delta)

    # --- Helper: 2-opt local search polish (optimized) ---
    def two_opt_polish(p, cost, max_time):
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = calc_delta(i, j, p)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    # --- Helper: Frank-Wolfe with adaptive step size ---
    def frank_wolfe_init(max_time):
        # Start with rank matching
        heavy_order = np.argsort(-a_row_sums, kind='stable')
        central_order = np.argsort(b_row_sums, kind='stable')
        p_init = np.empty(n, dtype=np.int64)
        p_init[heavy_order] = central_order
        
        # Polish the initial permutation
        p_init, cost_init = two_opt_polish(p_init.copy(), calc_cost_full(p_init), min(max_time, time.time() + 5.0))
        
        # Initialize X as a permutation matrix for p_init
        X = np.zeros((n, n), dtype=np.float64)
        X[np.arange(n), p_init] = 1.0
        
        best_p = p_init.copy()
        best_cost = cost_init
        
        t = 0
        prev_cost = None
        while time.time() < max_time:
            # Compute Gradient
            if sym:
                G = 2.0 * (a @ X @ b)
            else:
                G = (a @ X @ b.T) + (a.T @ X @ b)
            
            # Linear Assignment Problem (LAP) to find vertex P of Birkhoff polytope
            row_ind, col_ind = linear_sum_assignment(G)
            P = np.zeros((n, n), dtype=np.float64)
            P[row_ind, col_ind] = 1.0
            
            # Adaptive step size: start large, decay based on progress
            if t < 5:
                gamma = 0.8
            elif t < 20:
                gamma = 0.4
            elif t < 50:
                gamma = 0.2
            else:
                gamma = max(0.05, 2.0 / (t + 2.0))
            
            X = (1.0 - gamma) * X + gamma * P
            
            # Extract permutation from LAP solution
            p_candidate = col_ind.copy()
            
            # Polish the candidate with a quick 2-opt pass
            p_candidate, cost_candidate = two_opt_polish(p_candidate, calc_cost_full(p_candidate), min(max_time, time.time() + 1.5))
            
            if cost_candidate < best_cost:
                best_cost = cost_candidate
                best_p = p_candidate.copy()
            
            t += 1
            if time.time() > max_time:
                break

        return best_p, best_cost

    # --- Helper: Multi-start using linearized cost matrix ---
    def multi_start_search(max_time, num_starts):
        best_p_local = None
        best_cost_local = np.inf
        
        heavy_order = np.argsort(-a_row_sums, kind='stable')
        central_order = np.argsort(b_row_sums, kind='stable')
        
        for s in range(num_starts):
            if time.time() > max_time:
                break
            
            # Generate diverse initial permutations
            if s == 0:
                # Rank matching
                p_s = np.empty(n, dtype=np.int64)
                p_s[heavy_order] = central_order
            elif s == 1:
                # Reverse rank matching
                p_s = np.empty(n, dtype=np.int64)
                p_s[heavy_order] = central_order[::-1]
            elif s == 2:
                # Random permutation
                p_s = np.random.permutation(n).astype(np.int64)
            else:
                # Perturb best so far
                if best_p_local is not None:
                    p_s = best_p_local.copy()
                    # Random swaps
                    for _ in range(n // 4):
                        i, j = np.random.choice(n, 2, replace=False)
                        p_s[i], p_s[j] = p_s[j], p_s[i]
                else:
                    p_s = np.random.permutation(n).astype(np.int64)
            
            # Polish
            p_s, cost_s = two_opt_polish(p_s, calc_cost_full(p_s), min(max_time, time.time() + 8.0))
            
            if cost_s < best_cost_local:
                best_cost_local = cost_s
                best_p_local = p_s.copy()
        
        return best_p_local, best_cost_local

    # --- Strategy ---
    # 1. Quick rank matching init + polish
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    p, cost = two_opt_polish(p, calc_cost_full(p), time.time() + 8.0)
    best_p = p.copy()
    best_cost = cost

    # 2. Multi-start search with diverse initializations
    ms_deadline = time.time() + 25.0
    p_ms, cost_ms = multi_start_search(ms_deadline, num_starts=4)
    
    if cost_ms < best_cost:
        best_cost = cost_ms
        best_p = p_ms.copy()

    # 3. Run Frank-Wolfe
    fw_deadline = time.time() + 40.0
    p_fw, cost_fw = frank_wolfe_init(fw_deadline)
    
    if cost_fw < best_cost:
        best_cost = cost_fw
        best_p = p_fw.copy()

    # 4. Polish the best result found so far
    p_polish = best_p.copy()
    cost_polish = best_cost
    p_polish, cost_polish = two_opt_polish(p_polish, cost_polish, time.time() + 15.0)
    
    if cost_polish < best_cost:
        best_cost = cost_polish
        best_p = p_polish.copy()

    # 5. Simulated Annealing with improved schedule
    if time.time() < deadline - 5.0:
        p_sa = best_p.copy()
        cost_sa = best_cost
        
        # Better SA parameters
        T = 50.0
        T_min = 0.5
        alpha = 0.998
        no_improve = 0
        max_no_improve = 3000
        restart_count = 0
        max_restarts = 5
        
        while time.time() < deadline - 5.0:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            delta = calc_delta(i, j, p_sa)
            
            # Acceptance probability
            if delta < 0:
                accept = True
            elif T > T_min:
                accept = np.random.rand() < np.exp(-delta / T)
            else:
                accept = False
            
            if accept:
                p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                cost_sa += delta
                
                if cost_sa < best_cost:
                    best_cost = cost_sa
                    best_p = p_sa.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve > max_no_improve:
                    # Restart with higher temperature
                    restart_count += 1
                    if restart_count > max_restarts:
                        break
                    T = max(T * 2.0, 100.0)
                    no_improve = 0
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    restart_count += 1
                    if restart_count > max_restarts:
                        break
                    T = max(T * 2.0, 100.0)
                    no_improve = 0
                    
            T = max(T * alpha, T_min)

    # 6. Final Polish
    p_final = best_p.copy()
    cost_final = best_cost
    p_final, cost_final = two_opt_polish(p_final, cost_final, deadline)
    
    if cost_final < best_cost:
        best_cost = cost_final
        best_p = p_final.copy()

    return best_p.tolist()

# EVOLVE-BLOCK-END
