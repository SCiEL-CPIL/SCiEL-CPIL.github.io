"""QAP solver using MCMC with block-reversal moves (Gibbs/Metropolis-Hastings)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys


def _qap_cost(a, b, p):
    """Compute exact QAP cost for assignment p (facility i at location p[i])."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute change in QAP cost when swapping locations of facilities i and j.
    O(n) complexity.
    """
    if i == j:
        return 0.0
    
    pi = p[i]
    pj = p[j]
    
    # The change in cost involves:
    # 1. The pair (i, j) itself: A[i,j]*(B[pj, pi] - B[pi, pj])
    # 2. Pairs (i, k) and (j, k) for k != i, j
    # 3. Pairs (k, i) and (k, j) for k != i, j
    
    # Let's derive the O(n) formula.
    # Cost = sum_{u,v} A[u,v] B[p[u], p[v]]
    # Delta = Cost_new - Cost_old
    # Only p[i] and p[j] change.
    # Terms involving i or j change.
    
    # Sum over all k != i, j:
    # Old: A[i,k]B[pi, p[k]] + A[k,i]B[p[k], pi] + A[j,k]B[pj, p[k]] + A[k,j]B[p[k], pj]
    # New: A[i,k]B[pj, p[k]] + A[k,i]B[p[k], pj] + A[j,k]B[pi, p[k]] + A[k,j]B[p[k], pi]
    
    # Delta_k = A[i,k](B[pj, p[k]] - B[pi, p[k]]) + A[k,i](B[p[k], pj] - B[p[k], pi]) 
    #          + A[j,k](B[pi, p[k]] - B[pj, p[k]]) + A[k,j](B[p[k], pi] - B[p[k], pj])
    
    # Group by B terms:
    # B[pj, p[k]]: (A[i,k] - A[j,k])
    # B[pi, p[k]]: (A[j,k] - A[i,k]) = -(A[i,k] - A[j,k])
    # B[p[k], pj]: (A[k,i] - A[k,j])
    # B[p[k], pi]: (A[k,j] - A[k,i]) = -(A[k,i] - A[k,j])
    
    # So Delta_k = (A[i,k] - A[j,k]) * (B[pj, p[k]] - B[pi, p[k]]) 
    #            + (A[k,i] - A[k,j]) * (B[p[k], pj] - B[p[k], pi])
    
    # Plus the i-j term:
    # Old: A[i,j]B[pi, pj] + A[j,i]B[pj, pi]
    # New: A[i,j]B[pj, pi] + A[j,i]B[pi, pj]
    # Delta_ij = A[i,j](B[pj, pi] - B[pi, pj]) + A[j,i](B[pi, pj] - B[pj, pi])
    #          = (A[i,j] - A[j,i]) * (B[pj, pi] - B[pi, pj])
    
    # Total Delta = Delta_ij + sum_{k != i, j} Delta_k
    
    # We can compute the sum over all k and subtract k=i and k=j terms.
    # But k=i and k=j are handled in Delta_ij?
    # Let's check k=i in the sum formula:
    # (A[i,i] - A[j,i]) * (B[pj, pi] - B[pi, pi]) + (A[i,i] - A[i,j]) * (B[pi, pj] - B[pi, pi])
    # This is messy. Better to just sum over k != i, j.
    
    # Efficient NumPy implementation:
    # Create vectors for k in 0..n-1
    # Mask out i and j
    
    n = len(p)
    # We can compute for all k and then subtract the i and j contributions
    # which we calculate separately or just mask.
    
    # Vectorized sum for all k:
    # Diff_A_row = A[i, :] - A[j, :]
    # Diff_A_col = A[:, i] - A[:, j]
    # Term1 = Diff_A_row * (B[pj, p] - B[pi, p])  # Note: B[pj, p] is a row vector
    # Term2 = Diff_A_col * (B[p, pj] - B[p, pi])  # Note: B[p, pj] is a column vector
    
    # Wait, B is symmetric in QAP? No, usually not.
    # B[pj, p] means row pj of B, indexed by p.
    # B[p, pj] means column pj of B, indexed by p.
    
    # Let's just use the mask to be safe and clear, n=256 is small.
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    k_indices = np.where(mask)[0]
    
    if len(k_indices) == 0:
        return 0.0
        
    # Extract relevant rows/cols
    # B[pj, p[k]] - B[pi, p[k]]
    diff_b_row = b[pj, p[k_indices]] - b[pi, p[k_indices]]
    # B[p[k], pj] - B[p[k], pi]
    diff_b_col = b[p[k_indices], pj] - b[p[k_indices], pi]
    
    # A[i, k] - A[j, k]
    diff_a_row = a[i, k_indices] - a[j, k_indices]
    # A[k, i] - A[k, j]
    diff_a_col = a[k_indices, i] - a[k_indices, j]
    
    delta_sum = np.sum(diff_a_row * diff_b_row) + np.sum(diff_a_col * diff_b_col)
    
    # Delta_ij
    delta_ij = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return delta_sum + delta_ij


def _delta_cost_block_reverse(a, b, p, i, j):
    """
    Compute the change in QAP cost when we reverse the sub-sequence p[i..j].
    """
    length = j - i + 1
    if length <= 1:
        return 0
    
    S = np.arange(i, j + 1)
    old_locs = p[S]
    new_locs = p[S][::-1]
    
    # Indices for not S
    mask = np.ones(len(p), dtype=bool)
    mask[S] = False
    not_S = np.where(mask)[0]
    
    delta = 0.0
    
    # Part 1: f1 in S, f2 in S
    A_SS = a[np.ix_(S, S)]
    B_old_SS = b[np.ix_(old_locs, old_locs)]
    B_new_SS = b[np.ix_(new_locs, new_locs)]
    delta += np.sum(A_SS * (B_new_SS - B_old_SS))
    
    if len(not_S) > 0:
        # Part 2: f1 in S, f2 not in S
        A_S_notS = a[np.ix_(S, not_S)]
        B_old_S_notS = b[np.ix_(old_locs, p[not_S])]
        B_new_S_notS = b[np.ix_(new_locs, p[not_S])]
        delta += np.sum(A_S_notS * (B_new_S_notS - B_old_S_notS))
        
        # Part 3: f1 not in S, f2 in S
        A_notS_S = a[np.ix_(not_S, S)]
        B_old_notS_S = b[np.ix_(p[not_S], old_locs)]
        B_new_notS_S = b[np.ix_(p[not_S], new_locs)]
        delta += np.sum(A_notS_S * (B_new_notS_S - B_old_notS_S))
        
    return delta


def solve(a, b):
    n = len(a)
    
    # Initial solution: Greedy heuristic
    # Sort facilities by total flow descending
    # Sort locations by total distance ascending
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Heuristic: Assign highest flow facility to lowest distance location
    # This is a simple 1D assignment heuristic
    fac_order = np.argsort(-flow_sums)
    loc_order = np.argsort(dist_sums)
    
    p = np.empty(n, dtype=np.int64)
    for f_idx, l_idx in zip(fac_order, loc_order):
        p[f_idx] = l_idx
        
    # Local improvement: try all swaps
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                d = _delta_swap(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    improved = True
        # If improved, restart loop (or just continue, but restarting ensures we find a local min)
        # For n=256, this is 256*255/2 = 32640 swaps. Fast.
        
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    time_limit = 115.0
    
    # MCMC parameters
    # Temperature estimation
    temp = 1000.0  # Start high
    
    rng = np.random.default_rng(42)
    
    # Run MCMC
    last_improve = start_time
    no_improve_limit = 5.0 # seconds
    
    while time.time() - start_time < time_limit:
        # Choose move
        if rng.random() < 0.5:
            # Swap
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            delta = _delta_swap(a, b, p, i, j)
            
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    last_improve = time.time()
        else:
            # Block Reversal
            # Choose length
            r = rng.random()
            if r < 0.5:
                length = int(rng.integers(2, 10))
            elif r < 0.8:
                length = int(rng.integers(10, 50))
            else:
                length = int(rng.integers(50, min(129, n)))
                
            if length >= n:
                length = n - 1
            i = int(rng.integers(0, n - length + 1))
            j = i + length - 1
            
            delta = _delta_cost_block_reverse(a, b, p, i, j)
            
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                p[i:j+1] = p[i:j+1][::-1]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    last_improve = time.time()
        
        # Cooling
        temp *= 0.9999
        if temp < 1.0:
            temp = 1.0
            
        # Restart logic
        if time.time() - last_improve > no_improve_limit:
            # Perturb best solution
            p = best_p.copy()
            current_cost = best_cost
            # Randomly swap a few elements
            for _ in range(5):
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n))
                p[i], p[j] = p[j], p[i]
            current_cost = _qap_cost(a, b, p)
            temp = 1000.0
            last_improve = time.time()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
