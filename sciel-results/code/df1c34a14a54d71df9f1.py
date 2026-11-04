import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # --- Phase 1: Constructive Heuristic (Karmarkar-Karp style) ---
    # M[i, j] is the marginal cost of assigning facility i to location j,
    # given the current partial assignment.
    
    # Initialize M with the self-term: A[i,i] * B[j,j]
    M = np.outer(np.diag(a), np.diag(b))
    
    F = list(range(n))
    L = list(range(n))
    p = np.full(n, -1, dtype=np.int64)
    F_set = set(range(n))
    L_set = set(range(n))
    
    for step in range(n):
        if time.time() > deadline - 30.0:  # Leave 30s for local search
            break
            
        # Find min marginal cost among unassigned using numpy
        # Mask out assigned facilities and locations
        mask_i = np.ones(n, dtype=bool)
        mask_j = np.ones(n, dtype=bool)
        mask_i[p != -1] = False
        # L contains only unassigned locations
        for j in L:
            mask_j[j] = True
        for j in range(n):
            if p[j] != -1 or j not in L_set:
                mask_j[j] = False
        
        # Actually simpler: just use the current F and L lists
        if len(F) == 0 or len(L) == 0:
            break
            
        F_arr = np.array(F)
        L_arr = np.array(L)
        sub_M = M[np.ix_(F_arr, L_arr)]
        
        min_idx = np.unravel_index(np.argmin(sub_M), sub_M.shape)
        best_i = F_arr[min_idx[0]]
        best_j = L_arr[min_idx[1]]
        
        p[best_i] = best_j
        F.remove(best_i)
        L.remove(best_j)
        F_set.discard(best_i)
        L_set.discard(best_j)
        
        # Update M for remaining unassigned
        # M[i', j'] += A[i', best_i] * B[j', best_j] + A[best_i, i'] * B[best_j, j']
        if F and L:
            rem_i = np.array(F)
            rem_j = np.array(L)
            
            v1 = a[rem_i, best_i]
            v2 = b[best_j, rem_j]
            w1 = a[best_i, rem_i]
            w2 = b[rem_j, best_j]
            
            M[np.ix_(rem_i, rem_j)] += np.outer(v1, v2) + np.outer(w1, w2)
            
    # Complete permutation if KK timed out or failed
    if -1 in p:
        assigned_locs = set(p[p != -1])
        unassigned_locs = [j for j in range(n) if j not in assigned_locs]
        unassigned_fac = [i for i in range(n) if p[i] == -1]
        for i, j in zip(unassigned_fac, unassigned_locs):
            p[i] = j

    # --- Phase 2: Local Search with O(1) Delta Calculation ---
    
    # Precompute matrices for fast delta evaluation
    # R[i, l] = sum_k A[i, k] * B[l, p[k]]
    # C[i, l] = sum_k A[k, i] * B[p[k], l]
    
    def compute_precomp(p):
        B_col_p = b[:, p]
        B_row_p = b[p, :]
        
        R = a @ B_col_p.T
        C = a.T @ B_row_p
        
        return R, C

    R, C = compute_precomp(p)
    current_cost = int(np.sum(a * b[np.ix_(p, p)]))
    
    def calc_swap_delta(i, k, R, C, p, a, b):
        """Calculate delta for swapping positions i and k."""
        j = p[i]
        l = p[k]
        
        if j == l:
            return 0
        
        # Self terms
        d_self = (a[i, k] * b[l, j] + a[k, i] * b[j, l] + a[i, i] * b[l, l] + a[k, k] * b[j, j]) - \
                 (a[i, k] * b[j, l] + a[k, i] * b[l, j] + a[i, i] * b[j, j] + a[k, k] * b[l, l])
        
        # Cross terms with other facilities
        term_Ri = (R[i, l] - R[i, j]) - a[i,i]*(b[l, j] - b[j, j]) - a[i,k]*(b[l, l] - b[j, l])
        term_Rk = (R[k, j] - R[k, l]) - a[k,k]*(b[j, j] - b[l, l]) - a[k,i]*(b[j, l] - b[l, j])
        
        term_Ci = (C[i, l] - C[i, j]) - a[i,i]*(b[j, l] - b[j, j]) - a[k,i]*(b[l, l] - b[l, j])
        term_Ck = (C[k, j] - C[k, l]) - a[k,k]*(b[j, j] - b[l, l]) - a[i,k]*(b[j, l] - b[l, j])
        
        d_v = term_Ri + term_Rk + term_Ci + term_Ck
        
        return d_self + d_v
    
    def update_precomp_after_swap(i, k, R, C, p, a, b):
        """Update R and C after swapping p[i] and p[k]."""
        j = p[i]
        l = p[k]
        
        R += np.outer(a[:, i] - a[:, k], b[:, l] - b[:, j])
        C += np.outer(a[i, :] - a[k, :], b[l, :] - b[j, :])
        
        p[i], p[k] = p[k], p[i]
    
    # Phase 2a: Deterministic first pass - try all pairs in order
    # This helps escape poor initial solutions quickly
    for pass_num in range(3):  # Up to 3 full passes
        if time.time() > deadline - 15.0:
            break
        any_improvement = False
        for i in range(n):
            if time.time() > deadline - 15.0:
                break
            for k in range(i + 1, n):
                delta = calc_swap_delta(i, k, R, C, p, a, b)
                if delta < 0:
                    update_precomp_after_swap(i, k, R, C, p, a, b)
                    current_cost += delta
                    any_improvement = True
        if not any_improvement:
            break
    
    # Phase 2b: Randomized local search
    no_improve_count = 0
    max_no_improve = 20000
    np.random.seed(42)  # Reproducibility
    
    while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
        i = np.random.randint(0, n)
        k = np.random.randint(0, n)
        
        if i == k:
            continue
        
        delta = calc_swap_delta(i, k, R, C, p, a, b)
        
        if delta < 0:
            update_precomp_after_swap(i, k, R, C, p, a, b)
            current_cost += delta
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Phase 2c: Final polish - another deterministic pass
    if time.time() < deadline - 2.0:
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            for k in range(i + 1, n):
                if time.time() > deadline - 2.0:
                    break
                delta = calc_swap_delta(i, k, R, C, p, a, b)
                if delta < 0:
                    update_precomp_after_swap(i, k, R, C, p, a, b)
                    current_cost += delta

    return p.tolist()
