"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 118s
    deadline = time.time() + 118.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        diff_a_rows = a[i, :] - a[j, :]
        diff_b_rows = b[v, p] - b[u, p]
        s1 = np.dot(diff_a_rows, diff_b_rows)
        
        diff_a_cols = a[:, i] - a[:, j]
        diff_b_cols = b[p, v] - b[p, u]
        s2 = np.dot(diff_a_cols, diff_b_cols)
        
        ai_i = a[i, i]
        aj_i = a[j, i]
        ai_j = a[i, j]
        aj_j = a[j, j]
        
        bu_u = b[u, u]
        bu_v = b[u, v]
        bv_u = b[v, u]
        bv_v = b[v, v]
        
        corr_1 = (ai_i - aj_i) * (bv_u - bu_u)
        corr_2 = (ai_i - aj_i) * (bu_v - bu_u)
        corr_3 = (ai_j - aj_j) * (bv_v - bu_v)
        corr_4 = (ai_j - aj_j) * (bu_v - bv_v)
        
        corr_interaction = (ai_j - aj_i) * (bv_u - bu_v)
        
        return s1 + s2 - (corr_1 + corr_2 + corr_3 + corr_4) + corr_interaction

    def local_search(p, current_cost, deadline):
        improved = True
        max_iter = 300
        iter_count = 0
        
        while improved and time.time() < deadline - 3.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n - 1):
                if time.time() >= deadline - 3.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() >= deadline - 3.0:
                        return p, current_cost
                    d = swap_delta(i, j, p)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                iter_count += 1
            else:
                break
                
        return p, current_cost

    def generate_inits(a, b, deadline):
        inits = []
        rng = np.random.RandomState(12345)
        
        # 1. Rank-based initializations
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        
        p_rank = np.empty(n, dtype=np.int64)
        p_rank[heavy] = central
        inits.append(p_rank)
        
        p_rank_rev = np.empty(n, dtype=np.int64)
        p_rank_rev[heavy] = central[::-1]
        inits.append(p_rank_rev)
        
        # 2. Hungarian-based initializations
        # Heuristic: Minimize sum_i,j a_ij b_p(i)p(j)
        # Approximate cost for assigning i to j:
        # C_ij = (sum_k a_ik) * (sum_l b_jl)
        # This is O(n^2) to compute the matrix C.
        
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        
        # Outer product
        C1 = np.outer(a_row_sums, b_row_sums)
        
        try:
            if time.time() < deadline - 10.0:
                row_ind, col_ind = linear_sum_assignment(C1)
                p_hung1 = np.empty(n, dtype=np.int64)
                p_hung1[row_ind] = col_ind
                inits.append(p_hung1)
                
                # Reverse logic for minimization? 
                # linear_sum_assignment minimizes the sum of C1[row, col].
                # If we want to align high a_row_sums with low b_row_sums (to minimize product?),
                # the outer product C1 has large values for (high_a, high_b).
                # Minimizing sum C1[i, p[i]] will pair high a with low b if the matrix is monotonic?
                # Actually, if a_row_sums and b_row_sums are positive, minimizing sum a_i b_p(i) 
                # pairs largest a with smallest b. This is a valid heuristic.
                
                # Alternative: use column sums
                a_col_sums = a.sum(axis=0)
                b_col_sums = b.sum(axis=0)
                C2 = np.outer(a_col_sums, b_col_sums)
                
                row_ind2, col_ind2 = linear_sum_assignment(C2)
                p_hung2 = np.empty(n, dtype=np.int64)
                p_hung2[row_ind2] = col_ind2
                inits.append(p_hung2)
        except Exception:
            pass

        # 3. Spectral initialization
        try:
            if time.time() < deadline - 10.0:
                a_centered = a - a.mean()
                b_centered = b - b.mean()
                a_scale = np.sqrt(np.mean(a_centered ** 2)) + 1e-10
                b_scale = np.sqrt(np.mean(b_centered ** 2)) + 1e-10
                a_norm = a_centered / a_scale
                b_norm = b_centered / b_scale
                
                C_spec = 0.5 * (a_norm + a_norm.T) + 0.5 * (b_norm + b_norm.T)
                C_spec = 0.5 * (C_spec + C_spec.T)
                
                k = 5
                eigenvalues, eigenvectors = eigh(C_spec, subset_by_index=[n-k, n-1])
                
                for idx in range(k):
                    eigvec = eigenvectors[:, idx]
                    facility_order = np.argsort(eigvec, kind="stable")
                    location_order = np.argsort(-eigvec, kind="stable")
                    p = np.empty(n, dtype=np.int64)
                    p[facility_order] = location_order
                    inits.append(p)
        except Exception:
            pass

        # 4. Random initializations
        for _ in range(3):
            inits.append(rng.permutation(n))
            
        return inits

    # Generate diverse initializations
    inits = generate_inits(a, b, deadline)
    
    best_cost = np.inf
    best_p = None
    
    # Run local search from each initialization
    for p_init in inits:
        if time.time() > deadline - 15.0:
            break
        
        p_work = p_init.copy()
        cost = compute_cost(p_work)
        
        p_work, cost = local_search(p_work, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
    
    # Iterated Local Search (ILS) to escape local minima
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)
    
    # Adaptive perturbation: start small, increase if stuck
    base_perturb = 5
    max_perturb = 20
    no_improve_count = 0
    
    while time.time() < deadline - 8.0:
        # Perturb the best solution
        p = best_p.copy()
        
        # Adaptive perturbation size
        perturb_size = base_perturb + min(no_improve_count, 5) * 2
        perturb_size = min(perturb_size, max_perturb)
        
        num_swaps = np.random.randint(max(2, perturb_size // 2), perturb_size + 1)
        for _ in range(num_swaps):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = compute_cost(p)
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If we've tried many times without improvement, increase perturbation
        if no_improve_count > 10:
            no_improve_count = 0  # Reset to continue searching with current perturbation level
    
    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    return list(range(n))

# EVOLVE-BLOCK-END
