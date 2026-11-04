"""QAP solver for tho150 using Simulated Annealing with O(n) delta evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit
    
    # Convert to float64 for fast arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    # Precompute row sums of B for potential use, though delta formula is self-contained
    # b_row_sums = b_f.sum(axis=1)
    
    def compute_cost(p):
        """Compute exact cost using vectorized matrix multiplication."""
        return np.sum(a_f * b_f[np.ix_(p, p)])
    
    def spectral_init(a_mat, b_mat, dim=10):
        """
        Spectral initialization:
        1. Compute top eigenvectors of A (flow) and B (distance).
        2. Align them via Procrustes (rotation/reflection).
        3. Solve assignment problem on the aligned coordinates.
        """
        eigvals_a, eigvecs_a = np.linalg.eigh(a_mat)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_mat)
        
        # Take top dim eigenvectors
        idx_a = np.argsort(eigvals_a)[::-1][:dim]
        idx_b = np.argsort(eigvals_b)[::-1][:dim]
        
        X = eigvecs_a[:, idx_a] * np.sqrt(np.maximum(eigvals_a[idx_a], 0))
        Y = eigvecs_b[:, idx_b] * np.sqrt(np.maximum(eigvals_b[idx_b], 0))
        
        # Procrustes alignment: find R such that X @ R approx Y
        XtX = X.T @ X
        XtY = X.T @ Y
        try:
            R = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            R = np.linalg.lstsq(XtX, XtY, rcond=None)[0]
            
        # Align X to Y
        X_aligned = X @ R
        
        # Solve assignment: minimize sum ||x_i - y_{p[i]}||^2
        # Equivalent to maximizing sum x_i . y_{p[i]}
        D = X_aligned @ Y.T
        row_ind, col_ind = linear_sum_assignment(-D)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p

    from scipy.optimize import linear_sum_assignment
    
    # 1. Spectral Initialization
    p_curr = spectral_init(a_f, b_f, dim=10)
    curr_cost = compute_cost(p_curr)
    
    best_p = p_curr.copy()
    best_cost = curr_cost
    
    # 2. Simulated Annealing
    # Use O(n) delta calculation to allow many more iterations
    rng = np.random.RandomState(42)
    
    # Temperature schedule
    T_start = 5000.0
    T_end = 1.0
    # We don't know exactly how many steps we can take, so we estimate.
    # Assume ~1000 steps/sec -> ~140k steps in 140s.
    # Let's just run until time limit, adjusting T based on elapsed time.
    
    # To make SA effective, we need a good cooling schedule.
    # Let's estimate max steps based on a small trial run? No, just use time-based cooling.
    # Or fixed number of iterations if we can estimate speed.
    # Let's do a quick benchmark of 100 swaps to estimate speed.
    
    p_test = p_curr.copy()
    t0 = time.time()
    for _ in range(100):
        i, j = rng.choice(n, 2, replace=False)
        li = p_test[i]
        lj = p_test[j]
        if li != lj:
            # Just simulate the calculation cost, not actually swapping
            _ = 0.0 # Placeholder
    t1 = time.time()
    avg_time_per_step = (t1 - t0) / 100.0
    # This is very rough as we didn't do the full delta calc.
    # Let's just use a dynamic T that decays exponentially with elapsed time.
    # T(t) = T_start * (T_end/T_start)^(t/t_total)
    
    # We'll use a simpler approach: run for a fixed number of "epochs" 
    # where each epoch consists of N swaps, and cool down after each epoch.
    # But we don't know N.
    
    # Alternative: Just use time-based cooling directly.
    # T = T_start * exp(-k * elapsed)
    # Choose k such that T ~ T_end at deadline.
    # T_end = T_start * exp(-k * 148)
    # k = -ln(T_end/T_start) / 148
    
    k_cool = -np.log(T_end / T_start) / 148.0
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        T = T_start * np.exp(-k_cool * elapsed)
        
        # Randomly select two facilities to swap
        i, j = rng.choice(n, 2, replace=False)
        
        li = p_curr[i]
        lj = p_curr[j]
        
        if li == lj:
            continue
            
        # Compute delta cost for swapping locations of i and j
        # Cost = sum_{k,l} a[k,l] b[p[k], p[l]]
        # Change involves terms where k or l is i or j.
        
        # Let's derive the delta formula carefully.
        # Original terms involving i or j:
        # 1. k=i, l=j: a[i,j] b[li, lj]
        # 2. k=i, l=k' (k' != i,j): a[i,k'] b[li, p[k']]
        # 3. k=k' (k' != i,j), l=i: a[k',i] b[p[k'], li]
        # 4. k=j, l=i: a[j,i] b[lj, li]
        # 5. k=j, l=k' (k' != i,j): a[j,k'] b[lj, p[k']]
        # 6. k=k' (k' != i,j), l=j: a[k',j] b[p[k'], lj]
        # 7. k=i, l=i: a[i,i] b[li, li]
        # 8. k=j, l=j: a[j,j] b[lj, lj]
        
        # New terms after swap (i gets lj, j gets li):
        # 1. k=i, l=j: a[i,j] b[lj, li]
        # 2. k=i, l=k': a[i,k'] b[lj, p[k']]
        # 3. k=k', l=i: a[k',i] b[p[k'], lj]
        # 4. k=j, l=i: a[j,i] b[li, lj]
        # 5. k=j, l=k': a[j,k'] b[li, p[k']]
        # 6. k=k', l=j: a[k',j] b[p[k'], li]
        # 7. k=i, l=i: a[i,i] b[lj, lj]
        # 8. k=j, l=j: a[j,j] b[li, li]
        
        # Delta = New - Old
        
        # Vectorized calculation for terms involving k' (all k' != i, j)
        # Let mask be all indices except i and j.
        # We can compute sums over all k' and subtract the i, j parts if we include them, 
        # or just compute over all and adjust.
        # Let's compute sums over ALL k' including i and j, then subtract the specific i,j terms we handled separately?
        # No, it's cleaner to just sum over all k and handle the i,j cross terms carefully.
        
        # Let's use the formula:
        # Delta = sum_{k != i, j} a[i, k] (b[lj, p[k]] - b[li, p[k]])
        #       + sum_{k != i, j} a[k, i] (b[p[k], lj] - b[p[k], li])
        #       + sum_{k != i, j} a[j, k] (b[li, p[k]] - b[lj, p[k]])
        #       + sum_{k != i, j} a[k, j] (b[p[k], li] - b[p[k], lj])
        #       + (terms where k=i or j)
        
        # Let's compute the sums over all k first, then subtract k=i and k=j contributions.
        
        # Sum1 = sum_k a[i, k] b[lj, p[k]]
        # Sum2 = sum_k a[i, k] b[li, p[k]]
        # Sum3 = sum_k a[k, i] b[p[k], lj]
        # Sum4 = sum_k a[k, i] b[p[k], li]
        # Sum5 = sum_k a[j, k] b[li, p[k]]
        # Sum6 = sum_k a[j, k] b[lj, p[k]]
        # Sum7 = sum_k a[k, j] b[p[k], li]
        # Sum8 = sum_k a[k, j] b[p[k], lj]
        
        # These are dot products.
        # b_lj_pk = b_f[lj, p_curr]
        # b_li_pk = b_f[li, p_curr]
        # b_pk_lj = b_f[p_curr, lj]
        # b_pk_li = b_f[p_curr, li]
        
        b_lj_pk = b_f[lj, p_curr]
        b_li_pk = b_f[li, p_curr]
        b_pk_lj = b_f[p_curr, lj]
        b_pk_li = b_f[p_curr, li]
        
        S1 = np.dot(a_f[i, :], b_lj_pk)
        S2 = np.dot(a_f[i, :], b_li_pk)
        S3 = np.dot(a_f[:, i], b_pk_lj)
        S4 = np.dot(a_f[:, i], b_pk_li)
        S5 = np.dot(a_f[j, :], b_li_pk)
        S6 = np.dot(a_f[j, :], b_lj_pk)
        S7 = np.dot(a_f[:, j], b_pk_li)
        S8 = np.dot(a_f[:, j], b_pk_lj)
        
        # The sums S1..S8 include terms where k=i and k=j.
        # We need to subtract the k=i and k=j terms from these sums to get the "k != i, j" part,
        # and then add back the correct "new" terms for k=i, j.
        
        # Actually, let's just compute the full delta directly including all terms.
        
        # Terms for k=i in S1: a[i, i] b[lj, li]
        # Terms for k=j in S1: a[i, j] b[lj, lj]
        
        # Terms for k=i in S2: a[i, i] b[li, li]
        # Terms for k=j in S2: a[i, j] b[li, lj]
        
        # Terms for k=i in S3: a[i, i] b[li, lj]  (Note: p[i]=li, so b[p[i], lj] = b[li, lj])
        # Terms for k=j in S3: a[j, i] b[lj, lj]  (Note: p[j]=lj, so b[p[j], lj] = b[lj, lj])
        
        # Terms for k=i in S4: a[i, i] b[li, li]
        # Terms for k=j in S4: a[j, i] b[lj, li]
        
        # Terms for k=i in S5: a[j, i] b[li, li]
        # Terms for k=j in S5: a[j, j] b[li, lj]
        
        # Terms for k=i in S6: a[j, i] b[lj, li]
        # Terms for k=j in S6: a[j, j] b[lj, lj]
        
        # Terms for k=i in S7: a[i, j] b[li, li]  (Note: p[i]=li, so b[p[i], li] = b[li, li])
        # Terms for k=j in S7: a[j, j] b[lj, li]  (Note: p[j]=lj, so b[p[j], li] = b[lj, li])
        
        # Terms for k=i in S8: a[i, j] b[li, lj]
        # Terms for k=j in S8: a[j, j] b[lj, lj]
        
        # Let's define the "Old" contribution of the specific i,j interactions that change.
        # It is easier to compute:
        # Delta = (S1 - S2) + (S3 - S4) + (S5 - S6) + (S7 - S8)
        # This formula assumes that for k != i, j, the terms are correct.
        # For k=i and k=j, let's verify.
        
        # Let's check the k=i terms in (S1 - S2) + (S3 - S4) + (S5 - S6) + (S7 - S8):
        # From S1-S2: a[i,i](b[lj,li] - b[li,li])
        # From S3-S4: a[i,i](b[li,lj] - b[li,li])
        # From S5-S6: a[j,i](b[li,li] - b[lj,li])
        # From S7-S8: a[i,j](b[li,li] - b[li,lj])
        
        # Sum for k=i:
        # a[i,i](b[lj,li] - b[li,li] + b[li,lj] - b[li,li]) + a[j,i](b[li,li] - b[lj,li]) + a[i,j](b[li,li] - b[li,lj])
        
        # Now let's look at the actual change in cost for terms involving i (where i is k or l).
        # Old cost terms involving i:
        # k=i, l=i: a[i,i]b[li,li]
        # k=i, l=j: a[i,j]b[li,lj]
        # k=j, l=i: a[j,i]b[lj,li]
        # k=k', l=i: a[k',i]b[p[k'],li]
        
        # New cost terms involving i (i is now at lj):
        # k=i, l=i: a[i,i]b[lj,lj]
        # k=i, l=j: a[i,j]b[lj,li]
        # k=j, l=i: a[j,i]b[li,lj]
        # k=k', l=i: a[k',i]b[p[k'],lj]
        
        # Change for k=i, l=i: a[i,i](b[lj,lj] - b[li,li])
        # Change for k=i, l=j: a[i,j](b[lj,li] - b[li,lj])
        # Change for k=j, l=i: a[j,i](b[li,lj] - b[lj,li])
        # Change for k=k', l=i: sum_{k'!=i,j} a[k',i](b[p[k'],lj] - b[p[k'],li])
        
        # This is getting complicated to verify term by term.
        # Let's stick to the vectorized sum formula which is standard and robust if derived correctly.
        # The formula Delta = (S1 - S2) + (S3 - S4) + (S5 - S6) + (S7 - S8)
        # where S1 = sum_k a[i,k]b[lj,p[k]], etc.
        
        # Let's re-verify the k=i term in this formula.
        # Formula contribution for k=i:
        # S1: a[i,i]b[lj,li]
        # S2: a[i,i]b[li,li]
        # S3: a[i,i]b[li,lj]
        # S4: a[i,i]b[li,li]
        # S5: a[j,i]b[li,li]
        # S6: a[j,i]b[lj,li]
        # S7: a[i,j]b[li,li]
        # S8: a[i,j]b[li,lj]
        
        # Total Formula k=i:
        # a[i,i](b[lj,li] - b[li,li] + b[li,lj] - b[li,li]) + a[j,i](b[li,li] - b[lj,li]) + a[i,j](b[li,li] - b[li,lj])
        
        # Actual Change k=i (from above):
        # a[i,i](b[lj,lj] - b[li,li]) + a[i,j](b[lj,li] - b[li,lj]) + a[j,i](b[li,lj] - b[lj,li]) + (k' terms)
        
        # They don't match. The formula (S1-S2)... includes interactions with j in the sums S1..S8,
        # which double counts or miscounts the i-j interaction.
        
        # Correct approach:
        # 1. Compute sum over k != i, j.
        # 2. Add exact change for k=i, j, l=i, j terms.
        
        # Let's compute sums over all k, then subtract k=i and k=j.
        
        # Sum_k_neq_ij (S1_part) = S1 - a[i,i]b[lj,li] - a[i,j]b[lj,lj]
        # Sum_k_neq_ij (S2_part) = S2 - a[i,i]b[li,li] - a[i,j]b[li,lj]
        # Sum_k_neq_ij (S3_part) = S3 - a[i,i]b[li,lj] - a[j,i]b[lj,lj]
        # Sum_k_neq_ij (S4_part) = S4 - a[i,i]b[li,li] - a[j,i]b[lj,li]
        # Sum_k_neq_ij (S5_part) = S5 - a[j,i]b[li,li] - a[j,j]b[li,lj]
        # Sum_k_neq_ij (S6_part) = S6 - a[j,i]b[lj,li] - a[j,j]b[lj,lj]
        # Sum_k_neq_ij (S7_part) = S7 - a[i,j]b[li,li] - a[j,j]b[lj,li]
        # Sum_k_neq_ij (S8_part) = S8 - a[i,j]b[li,lj] - a[j,j]b[lj,lj]
        
        # Delta_k_neq_ij = (S1' - S2') + (S3' - S4') + (S5' - S6') + (S7' - S8')
        
        # Now add the exact change for the 4 terms involving i and j directly (k in {i,j}, l in {i,j}).
        # Old:
        # C_ii = a[i,i]b[li,li]
        # C_ij = a[i,j]b[li,lj]
        # C_ji = a[j,i]b[lj,li]
        # C_jj = a[j,j]b[lj,lj]
        
        # New:
        # C'_ii = a[i,i]b[lj,lj]
        # C'_ij = a[i,j]b[lj,li]
        # C'_ji = a[j,i]b[li,lj]
        # C'_jj = a[j,j]b[li,li]
        
        # Delta_direct = (C'_ii - C_ii) + (C'_ij - C_ij) + (C'_ji - C_ji) + (C'_jj - C_jj)
        
        # Total Delta = Delta_k_neq_ij + Delta_direct
        
        # Let's implement this.
        
        S1_sub = S1 - a_f[i, i] * b_f[lj, li] - a_f[i, j] * b_f[lj, lj]
        S2_sub = S2 - a_f[i, i] * b_f[li, li] - a_f[i, j] * b_f[li, lj]
        S3_sub = S3 - a_f[i, i] * b_f[li, lj] - a_f[j, i] * b_f[lj, lj]
        S4_sub = S4 - a_f[i, i] * b_f[li, li] - a_f[j, i] * b_f[lj, li]
        S5_sub = S5 - a_f[j, i] * b_f[li, li] - a_f[j, j] * b_f[li, lj]
        S6_sub = S6 - a_f[j, i] * b_f[lj, li] - a_f[j, j] * b_f[lj, lj]
        S7_sub = S7 - a_f[i, j] * b_f[li, li] - a_f[j, j] * b_f[lj, li]
        S8_sub = S8 - a_f[i, j] * b_f[li, lj] - a_f[j, j] * b_f[lj, lj]
        
        delta = (S1_sub - S2_sub) + (S3_sub - S4_sub) + (S5_sub - S6_sub) + (S7_sub - S8_sub)
        
        # Direct terms
        delta += a_f[i, i] * (b_f[lj, lj] - b_f[li, li])
        delta += a_f[i, j] * (b_f[lj, li] - b_f[li, lj])
        delta += a_f[j, i] * (b_f[li, lj] - b_f[lj, li])
        delta += a_f[j, j] * (b_f[li, li] - b_f[lj, lj])
        
        # Acceptance criterion
        if delta < 0:
            # Accept if cost decreases
            p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
            curr_cost += delta
            
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_p = p_curr.copy()
        else:
            # Accept with probability exp(-delta / T)
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.rand() < prob:
                    p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
                    curr_cost += delta
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
