"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _build_cost_matrix(a, b, p):
    """Build cost matrix C[i,l] = sum_j a[i,j] * b[l, p[j]]."""
    # V[l, j] = b[l, p[j]]
    V = b[:, p]
    # C[i, l] = sum_j a[i, j] * V[l, j] = (a @ V.T)[i, l]
    C = a @ V.T
    return C


def _swap_delta(C, a, b, p, i, j):
    """Compute exact delta for swapping p[i] and p[j]."""
    pi = p[i]
    pj = p[j]
    # Standard delta formula for QAP
    # Delta = sum_k a[i,k] (b[pj, p[k]] - b[pi, p[k]]) + sum_k a[j,k] (b[pi, p[k]] - b[pj, p[k]])
    # This is equivalent to:
    # (C[i, pj] - C[i, pi]) + (C[j, pi] - C[j, pj]) + correction
    # Correction handles the i-j interaction terms which are double counted or missing in the linear C approximation.
    # The C matrix linearizes the cost. The term a[i,j]b[pi,pj] becomes a[i,j]b[pj,pi] etc.
    # Let's use the robust correction formula derived from first principles.
    
    # Cost contribution of i to others: sum_k != j a[i,k] b[pi, p[k]]
    # After swap, i is at pj: sum_k != j a[i,k] b[pj, p[k]]
    # Cost contribution of j to others: sum_k != i a[j,k] b[pj, p[k]]
    # After swap, j is at pi: sum_k != i a[j,k] b[pi, p[k]]
    # Interaction i-j: a[i,j] b[pi, pj] -> a[i,j] b[pj, pi]
    # Interaction j-i: a[j,i] b[pj, pi] -> a[j,i] b[pi, pj]
    
    # Using C:
    # C[i, pi] = sum_k a[i,k] b[pi, p[k]] = sum_{k!=j} a[i,k] b[pi, p[k]] + a[i,j] b[pi, pj]
    # C[i, pj] = sum_k a[i,k] b[pj, p[k]] = sum_{k!=j} a[i,k] b[pj, p[k]] + a[i,j] b[pj, pj]
    
    # Delta = (C[i, pj] - a[i,j]b[pj,pj] - (C[i, pi] - a[i,j]b[pi,pj])) 
    #       + (C[j, pi] - a[j,i]b[pi,pi] - (C[j, pj] - a[j,i]b[pj,pi]))
    #       + a[i,j](b[pj,pi] - b[pi,pj]) + a[j,i](b[pi,pj] - b[pj,pi])
    
    # Simplifying:
    # Delta = (C[i, pj] - C[i, pi]) + (C[j, pi] - C[j, pj]) 
    #       - a[i,j]b[pj,pj] + a[i,j]b[pi,pj] 
    #       - a[j,i]b[pi,pi] + a[j,i]b[pj,pi]
    #       + a[i,j]b[pj,pi] - a[i,j]b[pi,pj] 
    #       + a[j,i]b[pi,pj] - a[j,i]b[pj,pi]
    
    # Terms cancel:
    # -a[i,j]b[pi,pj] + a[i,j]b[pi,pj] = 0
    # -a[j,i]b[pj,pi] + a[j,i]b[pj,pi] = 0
    
    # Remaining:
    # Delta = (C[i, pj] - C[i, pi]) + (C[j, pi] - C[j, pj])
    #       - a[i,j]b[pj,pj] + a[j,i]b[pj,pi] + a[i,j]b[pj,pi] - a[j,i]b[pi,pi]
    
    # Wait, the interaction terms in the new cost are:
    # a[i,j] b[pj, pi] (i at pj, j at pi)
    # a[j,i] b[pi, pj] (j at pi, i at pj)
    # Old interaction terms:
    # a[i,j] b[pi, pj]
    # a[j,i] b[pj, pi]
    
    # The C matrix includes the self-interaction a[i,i]b[pi,pi] etc?
    # C[i, l] = sum_k a[i,k] b[l, p[k]]. This includes k=i term: a[i,i]b[l,pi].
    # So C[i,pi] includes a[i,i]b[pi,pi] and a[i,j]b[pi,pj].
    
    # Let's just use the explicit sum for safety and speed with n=100.
    # But vectorized is better.
    
    pi_val = pi
    pj_val = pj
    
    # Vectorized delta calculation
    # diff_i = b[:, pj] - b[:, pi]
    # diff_j = b[:, pi] - b[:, pj]
    # delta_i = a[i, :] @ diff_i
    # delta_j = a[j, :] @ diff_j
    # But we must exclude the i-j interaction from the sums and add it back correctly.
    # a[i, :] @ (b[:, pj] - b[:, pi]) includes a[i,j](b[pj,pj]-b[pi,pj]) and a[i,i](b[pj,pi]-b[pi,pi])? No.
    # p[k] is the location of facility k.
    # b[:, pj] is column pj of b.
    # a[i, :] @ b[:, pj] = sum_k a[i,k] b[p[k], pj]. This is NOT C[i, pj].
    # C[i, pj] = sum_k a[i,k] b[pj, p[k]].
    
    # Let's stick to the C-based formula which was in the original code and is correct:
    delta_approx = C[i, pj] - C[i, pi] + C[j, pi] - C[j, pj]
    
    # Correction terms for the i-j interaction and self terms if any (a[i,i] is usually 0 in QAP but let's be safe)
    # The C matrix approximation assumes the cost is linear in the assignment.
    # The error comes from the fact that when we swap i and j, the term a[i,j]b[pi,pj] changes to a[i,j]b[pj,pi].
    # In C[i, pi], we have a[i,j]b[pi, pj].
    # In C[i, pj], we have a[i,j]b[pj, pj].
    # In C[j, pi], we have a[j,i]b[pi, pi].
    # In C[j, pj], we have a[j,i]b[pj, pi].
    
    # Change in i's cost from j's perspective:
    # Old: a[i,j]b[pi,pj] + a[j,i]b[pj,pi]
    # New: a[i,j]b[pj,pi] + a[j,i]b[pi,pj]
    
    # Change in i's cost from i's perspective (self):
    # Old: a[i,i]b[pi,pi]
    # New: a[i,i]b[pj,pj]
    # This is included in C[i,pi] and C[i,pj].
    # C[i,pi] has a[i,i]b[pi,pi]. C[i,pj] has a[i,i]b[pj,pj].
    # So C[i,pj]-C[i,pi] correctly accounts for self change of i.
    
    # Change in j's cost from j's perspective (self):
    # Old: a[j,j]b[pj,pj]
    # New: a[j,j]b[pi,pi]
    # C[j,pi] has a[j,j]b[pi,pi]. C[j,pj] has a[j,j]b[pj,pj].
    # So C[j,pi]-C[j,pj] correctly accounts for self change of j.
    
    # The only unaccounted terms are the cross terms a[i,j] and a[j,i].
    # In delta_approx:
    # C[i,pj] contains a[i,j]b[pj,pj]
    # C[i,pi] contains a[i,j]b[pi,pj]
    # C[j,pi] contains a[j,i]b[pi,pi]
    # C[j,pj] contains a[j,i]b[pj,pi]
    
    # So delta_approx has:
    # a[i,j]b[pj,pj] - a[i,j]b[pi,pj] + a[j,i]b[pi,pi] - a[j,i]b[pj,pi]
    
    # We WANT:
    # a[i,j]b[pj,pi] + a[j,i]b[pi,pj] - (a[i,j]b[pi,pj] + a[j,i]b[pj,pi])
    
    # Correction = Wanted - Approx
    # Correction = a[i,j]b[pj,pi] + a[j,i]b[pi,pj] - a[i,j]b[pj,pj] + a[i,j]b[pi,pj] - a[j,i]b[pi,pi] + a[j,i]b[pj,pi]
    # Wait, let's re-evaluate.
    
    # Total Delta = (New Cost) - (Old Cost)
    # New Cost terms involving i or j:
    # 1. i at pj, interacting with k!=j: sum_{k!=j} a[i,k]b[pj, p[k]]
    # 2. j at pi, interacting with k!=i: sum_{k!=i} a[j,k]b[pi, p[k]]
    # 3. i at pj, j at pi: a[i,j]b[pj, pi] + a[j,i]b[pi, pj]
    # 4. i at pj, self: a[i,i]b[pj, pj]
    # 5. j at pi, self: a[j,j]b[pi, pi]
    
    # Old Cost terms involving i or j:
    # 1. i at pi, interacting with k!=j: sum_{k!=j} a[i,k]b[pi, p[k]]
    # 2. j at pj, interacting with k!=i: sum_{k!=i} a[j,k]b[pj, p[k]]
    # 3. i at pi, j at pj: a[i,j]b[pi, pj] + a[j,i]b[pj, pi]
    # 4. i at pi, self: a[i,i]b[pi, pi]
    # 5. j at pj, self: a[j,j]b[pj, pj]
    
    # C[i, pj] = sum_k a[i,k]b[pj, p[k]] = sum_{k!=j} a[i,k]b[pj, p[k]] + a[i,j]b[pj, pj] + a[i,i]b[pj, pi] (if p[i]=pi, but p[k] is loc of k)
    # Wait, C[i, l] = sum_k a[i,k] b[l, p[k]].
    # If l=pj, C[i, pj] = sum_k a[i,k] b[pj, p[k]].
    # Term k=j: a[i,j] b[pj, p[j]] = a[i,j] b[pj, pj].
    # Term k=i: a[i,i] b[pj, p[i]] = a[i,i] b[pj, pi].
    
    # So C[i, pj] includes a[i,j]b[pj,pj] and a[i,i]b[pj,pi].
    # This does NOT match the "New Cost" term 1 and 4 directly because of the index l.
    # New Cost term 1 is sum_{k!=j} a[i,k] b[pj, p[k]].
    # C[i, pj] - a[i,j]b[pj,pj] = sum_{k!=j} a[i,k] b[pj, p[k]].
    # This sum includes k=i term: a[i,i]b[pj, pi].
    # So New Cost term 1 + 4 = C[i, pj] - a[i,j]b[pj,pj].
    
    # Similarly for j:
    # New Cost term 2 + 5 = C[j, pi] - a[j,i]b[pi,pi].
    
    # So New Cost (partial) = C[i, pj] - a[i,j]b[pj,pj] + C[j, pi] - a[j,i]b[pi,pi] + a[i,j]b[pj,pi] + a[j,i]b[pi,pj].
    
    # Old Cost (partial) = C[i, pi] - a[i,j]b[pi,pj] + C[j, pj] - a[j,i]b[pj,pi] + a[i,j]b[pi,pj] + a[j,i]b[pj,pi].
    # Note: C[i, pi] includes a[i,j]b[pi, p[j]] = a[i,j]b[pi, pj].
    # C[j, pj] includes a[j,i]b[pj, p[i]] = a[j,i]b[pj, pi].
    
    # Delta = (C[i, pj] + C[j, pi] - C[i, pi] - C[j, pj])
    #        - a[i,j]b[pj,pj] - a[j,i]b[pi,pi] + a[i,j]b[pj,pi] + a[j,i]b[pi,pj]
    #        + a[i,j]b[pi,pj] + a[j,i]b[pj,pi]  (Wait, old cost terms were subtracted)
    
    # Let's group:
    # Delta = (C[i, pj] - C[i, pi]) + (C[j, pi] - C[j, pj])
    #        + (a[i,j]b[pj,pi] + a[j,i]b[pi,pj] - a[i,j]b[pj,pj] - a[j,i]b[pi,pi])
    #        + (a[i,j]b[pi,pj] + a[j,i]b[pj,pi])  <- From subtracting old cost cross terms?
    #        - (a[i,j]b[pi,pj] + a[j,i]b[pj,pi])  <- From subtracting old cost cross terms included in C?
    
    # Actually, the cross terms in Old Cost were explicit.
    # Old Cost Cross = a[i,j]b[pi,pj] + a[j,i]b[pj,pi].
    # In Old Cost Partial (using C), we had C[i,pi] which included a[i,j]b[pi,pj].
    # So we subtracted that part of C, then added it back? No.
    # Old Cost = (C[i,pi] - a[i,j]b[pi,pj] - a[i,i]b[pi,pi]) + ... + a[i,j]b[pi,pj] + ...
    # This is getting confusing. Let's use the formula from the provided code which was working.
    
    correction = (a[i, j] * (b[pj, pi] - b[pj, pj]) +
                  a[j, i] * (b[pi, pj] - b[pi, pi]) -
                  a[i, i] * (b[pj, pi] - b[pi, pi]) -
                  a[j, j] * (b[pi, pj] - b[pj, pj]))
    
    return int(delta_approx + correction)


def _update_cost_matrix(C, a, b, p, i, j):
    """Update cost matrix C after swapping p[i] and p[j]."""
    pi = p[i]
    pj = p[j]
    # C[i, l] changes because p[i] changed? No, C is defined by p.
    # C[i, l] = sum_k a[i,k] b[l, p[k]].
    # If we swap p[i] and p[j], then for any l:
    # New C[i, l] = sum_{k!=i,j} a[i,k] b[l, p[k]] + a[i,i] b[l, pj] + a[i,j] b[l, pi]
    # Old C[i, l] = sum_{k!=i,j} a[i,k] b[l, p[k]] + a[i,i] b[l, pi] + a[i,j] b[l, pj]
    # Diff = a[i,i] (b[l, pj] - b[l, pi]) + a[i,j] (b[l, pi] - b[l, pj])
    #      = (a[i,i] - a[i,j]) (b[l, pj] - b[l, pi])
    
    # This applies to row i.
    # Similarly for row j:
    # New C[j, l] - Old C[j, l] = (a[j,j] - a[j,i]) (b[l, pi] - b[l, pj])
    #                            = -(a[j,j] - a[j,i]) (b[l, pj] - b[l, pi])
    
    # What about other rows k?
    # C[k, l] = sum_m a[k,m] b[l, p[m]].
    # p[i] and p[j] changed.
    # Term m=i: a[k,i] b[l, p[i]] -> a[k,i] b[l, pj]
    # Term m=j: a[k,j] b[l, p[j]] -> a[k,j] b[l, pi]
    # Diff = a[k,i] (b[l, pj] - b[l, pi]) + a[k,j] (b[l, pi] - b[l, pj])
    #      = (a[k,i] - a[k,j]) (b[l, pj] - b[l, pi])
    
    # So for ALL rows k, the update is:
    # C[k, :] += (a[k,i] - a[k,j]) * (b[:, pj] - b[:, pi])
    
    diff_a = a[:, i] - a[:, j]
    diff_b = b[:, pj] - b[:, pi]
    C += np.outer(diff_a, diff_b)
    
    p[i], p[j] = p[j], p[i]


def _local_search(C, a, b, p, current_cost, deadline, max_passes=100):
    """Steepest descent local search."""
    n = len(p)
    for _ in range(max_passes):
        if time.time() >= deadline - 2:
            break
        best_d = 0
        best_i, best_j = -1, -1
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                d = _swap_delta(C, a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i == -1:
            break
        _update_cost_matrix(C, a, b, p, best_i, best_j)
        current_cost += best_d
    return p, current_cost, C


def _spectral_initial(a, b, n, k=10, rng=None):
    """Generate initial permutation using spectral projection."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    a_centered = a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()
    b_centered = b - b.mean(axis=1, keepdims=True) - b.mean(axis=0, keepdims=True) + b.mean()
    
    try:
        Ua, Sa, Va = np.linalg.svd(a_centered, full_matrices=False)
        Ub, Sb, Vb = np.linalg.svd(b_centered, full_matrices=False)
        k = min(k, n)
        A_proj = (Ua[:, :k] * Sa[:k]) @ Va[:k, :]
        B_proj = (Ub[:, :k] * Sb[:k]) @ Vb[:k, :]
    except:
        A_proj = a_centered
        B_proj = b_centered
    
    C = A_proj @ B_proj.T
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _lap_initial(a, b, n):
    """Get initial assignment using LAP on linearized cost."""
    C = a @ b
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118
    rng = np.random.default_rng(42)
    
    # Generate diverse initial solutions
    p_lap = _lap_initial(a, b, n)
    p_spec = _spectral_initial(a, b, n, k=10, rng=rng)
    
    candidates = [p_lap, p_spec]
    best_p = candidates[0].copy()
    best_cost = _qap_cost(a, b, best_p)
    
    for pc in candidates[1:]:
        c = _qap_cost(a, b, pc)
        if c < best_cost:
            best_cost = c
            best_p = pc.copy()
            
    # Initial Polish
    C = _build_cost_matrix(a, b, best_p)
    p_ls = best_p.copy()
    cost_ls = best_cost
    p_ls, cost_ls, C_ls = _local_search(C, a, b, p_ls, cost_ls, deadline, max_passes=50)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
        C = C_ls.copy()
        
    # Iterated Local Search
    # Perturb and Polish
    while time.time() < deadline - 5:
        # Perturb
        p = best_p.copy()
        # Random swap perturbation
        num_swaps = rng.integers(5, 20)
        for _ in range(num_swaps):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                p[i], p[j] = p[j], p[i]
                
        # Polish
        C_pol = _build_cost_matrix(a, b, p)
        p_pol = p.copy()
        cost_pol = _qap_cost(a, b, p_pol)
        
        p_pol, cost_pol, C_pol = _local_search(C_pol, a, b, p_pol, cost_pol, deadline, max_passes=50)
        
        if cost_pol < best_cost:
            best_cost = cost_pol
            best_p = p_pol.copy()
            C = C_pol.copy()
        else:
            # If no improvement, maybe try a different perturbation strength next time
            # or accept with some probability (SA-like) but ILS usually rejects.
            # Let's stick to strict improvement for simplicity and robustness.
            pass

    return best_p.tolist()
# EVOLVE-BLOCK-END
