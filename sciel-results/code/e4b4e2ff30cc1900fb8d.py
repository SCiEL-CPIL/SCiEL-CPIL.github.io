"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _total_cost(a, b, p):
    """Compute the QAP cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_cost_delta(a, b, p, i, j):
    """Compute the cost change from swapping positions i and j in permutation p.
    p maps facility -> location. We swap p[i] and p[j].
    The cost is sum_{k,l} a[k,l] * b[p[k], p[l]].
    Changing p[i] and p[j] affects all terms where k in {i,j} or l in {i,j}.
    Terms affected: (i,i), (i,j), (j,i), (j,j) in the a-index, and for each of those,
    the b-index uses p[k] and p[l].
    """
    pi, pj = p[i], p[j]
    # Old contribution from rows i and j (all columns)
    # New contribution from rows i and j (all columns)
    # Also need to consider columns i and j for all rows? No.
    # The cost is sum_{k,l} a[k,l] * b[p[k], p[l]].
    # If we swap p[i] <-> p[j], then:
    #   For k=i: b[p[i], p[l]] changes for all l
    #   For k=j: b[p[j], p[l]] changes for all l
    #   For l=i: b[p[k], p[i]] changes for all k
    #   For l=j: b[p[k], p[j]] changes for all k
    # But the terms (i,i), (i,j), (j,i), (j,j) are counted in both row and column effects.
    # Let's compute the delta as:
    # new_cost - old_cost
    # = sum_l a[i,l]*(b[pj,p[l]] - b[pi,p[l]]) + sum_l a[j,l]*(b[pi,p[l]] - b[pj,p[l]])
    #   + sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi]) + sum_k a[k,j]*(b[p[k],pi] - b[p[k],pj])
    #   - corrections for double-counted terms (i,i),(i,j),(j,i),(j,j)
    # Actually, it's cleaner to just compute the full delta:
    # The affected b-indices are pi and pj.
    # Old: sum_{k,l} a[k,l] * b[p[k], p[l]]
    # The change only happens where p[k] in {pi,pj} or p[l] in {pi,pj}.
    # p[k] = pi iff k=i; p[k] = pj iff k=j.
    # p[l] = pi iff l=i; p[l] = pj iff l=j.
    # So affected terms are those with k in {i,j} OR l in {i,j}.
    # By inclusion-exclusion:
    # Sum over k in {i,j}, all l: a[k,l]*b[p[k],p[l]]
    # + Sum over l in {i,j}, all k not in {i,j}: a[k,l]*b[p[k],p[l]]
    # Old value of affected terms:
    old = (a[i, :] @ b[pi, p] + a[j, :] @ b[pj, p]
           + a[:, i] @ b[p, pi] + a[:, j] @ b[p, pj]
           - a[i, i] * b[pi, pi] - a[i, j] * b[pi, pj]
           - a[j, i] * b[pj, pi] - a[j, j] * b[pj, pj])
    # New value: p[i] becomes pj, p[j] becomes pi
    new = (a[i, :] @ b[pj, p] + a[j, :] @ b[pi, p]
           + a[:, i] @ b[p, pj] + a[:, j] @ b[p, pi]
           - a[i, i] * b[pj, pj] - a[i, j] * b[pj, pi]
           - a[j, i] * b[pi, pj] - a[j, j] * b[pi, pi])
    return new - old


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # --- Phase 1: Lagrangian-relaxation-inspired initialization ---
    # Use linear assignment on a transformed matrix to get a good starting point.
    # The QAP is: min sum_{i,j} a[i,j] b[p[i],p[j]]
    # A common relaxation: fix the "center of mass" of b.
    # We can try: for each facility i, the effective cost of assigning it to location l
    # is approximately sum_j a[i,j] * b[l, c(j)] where c is some reference assignment.
    # Start with rank-matching, then refine with a linear assignment.
    
    # Rank-matching initial guess
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Refine with a few rounds of linear assignment
    # In each round, for each facility i, compute cost vector c_i[l] = sum_j a[i,j] * b[l, p[j]]
    # Then solve the linear assignment problem.
    for _ in range(5):
        if time.time() > deadline - 60.0:
            break
        # Compute cost matrix C where C[i,l] = sum_j a[i,j] * b[l, p[j]]
        # C = b @ (a @ b[:, p]) ... let's think.
        # C[i,l] = sum_j a[i,j] * b[l, p[j]] = (a @ (b[:, p].T))[i, l]? No.
        # b[:, p] is an n x n matrix with columns permuted by p.
        # Let B_p = b[:, p] (n x n, column j is b[:, p[j]])
        # C[i,l] = sum_j a[i,j] * B_p[l,j] = (a @ B_p.T)[i,l]? 
        # (a @ B_p.T)[i,l] = sum_j a[i,j] * B_p.T[j,l] = sum_j a[i,j] * B_p[l,j]. Yes!
        B_p = b[:, p]
        C = a @ B_p.T
        # Solve linear assignment
        _, col_assign = linear_sum_assignment(C)
        # col_assign[i] is the location assigned to facility i
        p_new = col_assign.astype(np.int64)
        # Check if it's a valid permutation (it should be from LAP)
        if np.array_equal(np.sort(p_new), np.arange(n)):
            p = p_new
    
    best_p = p.copy()
    best_cost = _total_cost(a, b, p)
    
    # --- Phase 2: Vectorized Simulated Annealing with swap moves ---
    # Precompute some useful arrays
    # For fast swap delta computation, we precompute:
    # row_sums_a[i] = sum_l a[i,l]  (not directly useful)
    # We need: for swap (i,j), delta = new - old as computed above.
    # Let's precompute:
    # A_rows = a  (n x n)
    # B_cols_p = b[:, p]  (n x n) - updates when p changes
    # We'll maintain B_p = b[:, p] and update it on swaps.
    
    B_p = b[:, p].copy()  # n x n, column j is b[:, p[j]]
    
    # Precompute for faster delta:
    # For facility i: a[i, :] @ B_p = vector of length n, entry l = sum_j a[i,j]*b[p[j], l]? No.
    # a[i,:] @ B_p: entry l = sum_j a[i,j] * B_p[j,l] = sum_j a[i,j] * b[p[j], l]
    # That's not quite what we need. We need b[pi, p[l]] type terms.
    
    # Let me re-derive the swap delta more carefully and precompute what we can.
    # For swap (i, j) with pi = p[i], pj = p[j]:
    # delta = sum_l a[i,l]*(b[pj,p[l]] - b[pi,p[l]]) + sum_l a[j,l]*(b[pi,p[l]] - b[pj,p[l]])
    #       + sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi]) + sum_k a[k,j]*(b[p[k],pi] - b[p[k],pj])
    #       - [corrections for (i,i),(i,j),(j,i),(j,j) double counting]
    # 
    # Let's define:
    # R_i[l] = b[pi, p[l]] for all l  -> this is b[pi, p] which is row pi of b indexed by p
    # R_j[l] = b[pj, p[l]] for all l
    # C_i[k] = b[p[k], pi] for all k  -> this is b[p, pi] which is column pi of b indexed by p
    # C_j[k] = b[p[k], pj] for all k
    #
    # delta = a[i,:] @ (R_j - R_i) + a[j,:] @ (R_i - R_j) + a[:,i] @ (C_j - C_i) + a[:,j] @ (C_i - C_j)
    #       - a[i,i]*(b[pj,pj]-b[pi,pi]) - a[i,j]*(b[pj,pi]-b[pi,pj])
    #       - a[j,i]*(b[pi,pj]-b[pj,pi]) - a[j,j]*(b[pi,pi]-b[pj,pj])
    #
    # = (a[i,:] - a[j,:]) @ (R_j - R_i) + (a[:,i] - a[:,j]) @ (C_j - C_i)
    #   - a[i,i]*(b[pj,pj]-b[pi,pi]) - a[i,j]*(b[pj,pi]-b[pi,pj])
    #   - a[j,i]*(b[pi,pj]-b[pj,pi]) - a[j,j]*(b[pi,pi]-b[pj,pj])
    
    # For vectorized SA, we'll do random pairs and compute delta.
    # To speed up, we can precompute row and column slices.
    
    # SA parameters
    T = max(best_cost * 0.01, 1000.0)
    T_min = max(best_cost * 0.0001, 10.0)
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 5000
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 10.0:
        # Batch of swaps for vectorization
        batch_size = min(100, max(1, int((deadline - time.time()) * 100)))
        
        for _ in range(batch_size):
            if time.time() > deadline - 10.0:
                break
            
            # Random pair
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            pi = p[i]
            pj = p[j]
            
            # Compute delta
            # R_i = b[pi, p], R_j = b[pj, p]
            # C_i = b[p, pi], C_j = b[p, pj]
            R_diff = b[pj, p] - b[pi, p]  # n-vector
            C_diff = b[p, pj] - b[p, pi]  # n-vector
            
            a_row_diff = a[i, :] - a[j, :]
            a_col_diff = a[:, i] - a[:, j]
            
            delta = a_row_diff @ R_diff + a_col_diff @ C_diff
            # Corrections
            delta -= a[i, i] * (b[pj, pj] - b[pi, pi])
            delta -= a[i, j] * (b[pj, pi] - b[pi, pj])
            delta -= a[j, i] * (b[pi, pj] - b[pj, pi])
            delta -= a[j, j] * (b[pi, pi] - b[pj, pj])
            
            # Accept?
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                # Perform swap
                p[i], p[j] = pj, pi
                B_p[:, i], B_p[:, j] = B_p[:, j], B_p[:, i]
                
                new_cost = best_cost + delta
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_p = p.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                
                T = max(T * cooling, T_min)
        
        if no_improve > max_no_improve:
            # Kick: random perturbation
            for _ in range(n // 5):
                i = rng.integers(n)
                j = rng.integers(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
                    B_p[:, i], B_p[:, j] = B_p[:, j], B_p[:, i]
            no_improve = 0
            T = max(T * 2, T_min * 10)
    
    # Final polish: 2-opt local search
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 2.0:
                    break
                pi = p[i]
                pj = p[j]
                R_diff = b[pj, p] - b[pi, p]
                C_diff = b[p, pj] - b[p, pi]
                a_row_diff = a[i, :] - a[j, :]
                a_col_diff = a[:, i] - a[:, j]
                delta = a_row_diff @ R_diff + a_col_diff @ C_diff
                delta -= a[i, i] * (b[pj, pj] - b[pi, pi])
                delta -= a[i, j] * (b[pj, pi] - b[pi, pj])
                delta -= a[j, i] * (b[pi, pj] - b[pj, pi])
                delta -= a[j, j] * (b[pi, pi] - b[pj, pj])
                if delta < 0:
                    p[i], p[j] = pj, pi
                    B_p[:, i], B_p[:, j] = B_p[:, j], B_p[:, i]
                    best_cost += delta
                    best_p = p.copy()
                    improved = True
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
