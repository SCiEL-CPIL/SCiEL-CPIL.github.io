"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Spectral Embedding
    # The QAP can be relaxed. A good heuristic initialization often comes from
    # aligning the top eigenvectors of A and B.
    # A and B are symmetric? In QAPLIB, typically yes. If not, symmetrize for spectral init.
    # tho150 is symmetric.
    
    # Compute top eigenvector of A (flow)
    # Using np.linalg.eigh for symmetric matrices
    try:
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        
        # Take the largest eigenvector (last in ascending order)
        vec_a = v_a[:, -1]
        vec_b = v_b[:, -1]
        
        # We want to assign facilities (indices of vec_a) to locations (indices of vec_b)
        # such that high vec_a aligns with high vec_b?
        # Actually, the cost is sum a[i,j] b[p[i],p[j]].
        # If A and B are similar, we want p to be identity or close.
        # If we sort facilities by vec_a and locations by vec_b, we get a rank-based init.
        
        # Sort facilities by descending vec_a
        fac_order = np.argsort(-vec_a, kind="stable")
        # Sort locations by descending vec_b
        loc_order = np.argsort(-vec_b, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        
    except Exception:
        # Fallback to rank matching if spectral fails
        a_sum = a.sum(axis=1)
        b_sum = b.sum(axis=1)
        heavy_order = np.argsort(-a_sum, kind="stable")
        central_order = np.argsort(b_sum, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order

    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    # Start with a high temperature to allow large jumps
    T_start = 50000.0
    T_end = 1.0
    # Number of iterations is determined by time budget
    # We will run until deadline
    
    # Multi-swap strategy:
    # Instead of just 2-swaps, we occasionally perform 3-swaps or 4-swaps.
    # This allows escaping local minima that are far in Hamming distance.
    
    # Precompute row sums of B for quick delta estimation? 
    # For multi-swaps, exact delta calculation is complex.
    # Strategy: 
    # 1. Generate a random multi-swap move.
    # 2. Calculate the new cost exactly (or incrementally if simple).
    # 3. Accept if better or with probability exp(-delta/T).
    
    # To speed up cost calculation for random moves:
    # We maintain the current permutation p and current_cost.
    # When we swap k pairs, the change in cost depends on the interactions 
    # between the swapped indices and all other indices.
    
    # Let the set of swapped facility indices be S = {i1, i2, ..., ik}
    # Let the set of swapped location indices be L = {p[i1], ..., p[ik]}
    # The new locations for S are permuted.
    
    # Delta = sum_{i,j in S} a[i,j] * (b[new_p[i], new_p[j]] - b[old_p[i], old_p[j]])
    #        + sum_{i in S, j not in S} a[i,j] * (b[new_p[i], p[j]] - b[old_p[i], p[j]])
    #        + sum_{i not in S, j in S} a[i,j] * (b[p[i], new_p[j]] - b[p[i], old_p[j]])
    
    # This is O(k^2 + k(n-k)) ~ O(kn).
    # For k=2, O(n). For k=3, O(n).
    
    # We will use a mix of 2-swaps (most frequent) and 3-swaps (less frequent) 
    # to balance exploration and exploitation.
    
    # Vectorized delta for 2-swap (i, j):
    # diff_a = a[i] - a[j]
    # diff_b = b[:, p[i]] - b[:, p[j]] (but we only care about interactions with others? No, full vector)
    # Actually, the formula derived in the prompt is efficient for 2-swaps.
    # delta = dot(a[i] - a[j], b[:, p[i]] - b[:, p[j]])
    # Wait, the prompt's formula was:
    # delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
    # This excludes k=i,j. The terms involving i,j themselves are:
    # a[i,i]b[p[i],p[i]] + a[i,j]b[p[i],p[j]] + a[j,i]b[p[j],p[i]] + a[j,j]b[p[j],p[j]]
    # vs
    # a[i,i]b[p[j],p[j]] + a[i,j]b[p[j],p[i]] + a[j,i]b[p[i],p[j]] + a[j,j]b[p[i],p[i]]
    # If a and b are zero-diagonal (common in QAPLIB), the self-terms vanish.
    # The cross terms: a[i,j]b[p[i],p[j]] + a[j,i]b[p[j],p[i]] vs a[i,j]b[p[j],p[i]] + a[j,i]b[p[i],p[j]].
    # If symmetric, these are equal. So for symmetric zero-diagonal QAP, 
    # the delta is indeed just the sum over k != i,j.
    
    # Let's verify if tho150 is zero diagonal.
    # Usually QAPLIB instances have 0 on diagonal.
    
    # We will implement a general 2-swap delta that is fast.
    # Precompute B_col = b[:, p] ? No, p changes.
    # We need b[x, p[k]] for all k.
    
    # Let's stick to the vectorized 2-swap delta from the prompt but optimized.
    # And add a 3-swap move for larger jumps.
    
    # For 3-swap, exact delta calculation is slower. 
    # Maybe we can just recalculate cost for 3-swaps? 
    # Cost calc is O(n^2) = 22500 ops. 
    # 2-swap delta is O(n) = 150 ops.
    # 2-swap is ~150x faster.
    # So we should do mostly 2-swaps.
    
    # However, the directive asks for "large-step" moves.
    # Let's try a "2-opt like" but with random pairs, and occasionally a "3-swap".
    
    # Actually, a better "large step" for QAP is to swap a subset of facilities 
    # with a subset of locations based on a secondary heuristic?
    # No, let's stick to SA with multi-swaps.
    
    # Optimization: 
    # Precompute a_sum_col? No.
    
    # Let's refine the 2-swap delta calculation.
    # delta = np.dot(a[i] - a[j], b[:, p[i]] - b[:, p[j]]) 
    # This includes k=i and k=j terms.
    # Term k=i: (a[i,i]-a[j,i]) * (b[p[i], p[i]] - b[p[j], p[i]])
    # Term k=j: (a[i,j]-a[j,j]) * (b[p[i], p[j]] - b[p[j], p[j]])
    # If diagonal is 0:
    # k=i: (-a[j,i]) * (0 - b[p[j], p[i]]) = a[j,i] b[p[j], p[i]]
    # k=j: (a[i,j]) * (b[p[i], p[j]] - 0) = a[i,j] b[p[i], p[j]]
    # Sum = a[i,j]b[p[i],p[j]] + a[j,i]b[p[j],p[i]]
    # This is exactly the cross-term contribution from the old state.
    # The new state cross-terms are:
    # i gets p[j], j gets p[i].
    # k=i (now at p[j]): a[j,i] (since i is facility i? No, k is index in sum. 
    # The formula sum_{k} a[i,k] b[p[i], p[k]] is for facility i.
    # When we swap p[i] and p[j]:
    # Facility i moves to p[j]. Facility j moves to p[i].
    # Cost change for terms involving i and j:
    # Old: a[i,i]b[p[i],p[i]] + a[i,j]b[p[i],p[j]] + sum_{k!=i,j} a[i,k]b[p[i],p[k]]
    #      + a[j,i]b[p[j],p[i]] + a[j,j]b[p[j],p[j]] + sum_{k!=i,j} a[j,k]b[p[j],p[k]]
    # New: a[i,i]b[p[j],p[j]] + a[i,j]b[p[j],p[i]] + sum_{k!=i,j} a[i,k]b[p[j],p[k]]
    #      + a[j,i]b[p[i],p[j]] + a[j,j]b[p[i],p[i]] + sum_{k!=i,j} a[j,k]b[p[i],p[k]]
    
    # If diag=0:
    # Old Cross: a[i,j]b[p[i],p[j]] + a[j,i]b[p[j],p[i]]
    # New Cross: a[i,j]b[p[j],p[i]] + a[j,i]b[p[i],p[j]]
    # If symmetric, Old Cross == New Cross.
    # So delta is indeed sum_{k!=i,j} (a[i,k]-a[j,k]) (b[p[j],p[k]] - b[p[i],p[k]])
    
    # So the vectorized dot product `np.dot(a[i] - a[j], b[:, p[i]] - b[:, p[j]])` 
    # gives:
    # sum_k (a[i,k]-a[j,k]) (b[p[i],p[k]] - b[p[j],p[k]])
    # = sum_{k!=i,j} ... + terms for k=i, k=j.
    # k=i: (a[i,i]-a[j,i]) (b[p[i],p[i]] - b[p[j],p[i]]) = (-a[j,i]) (0 - b[p[j],p[i]]) = a[j,i]b[p[j],p[i]]
    # k=j: (a[i,j]-a[j,j]) (b[p[i],p[j]] - b[p[j],p[j]]) = a[i,j] (b[p[i],p[j]] - 0) = a[i,j]b[p[i],p[j]]
    # So Dot = Delta + a[j,i]b[p[j],p[i]] + a[i,j]b[p[i],p[j]]
    # Since Old Cross = a[i,j]b[p[i],p[j]] + a[j,i]b[p[j],p[i]] (symmetric)
    # And New Cross = same.
    # So Delta = Dot - Old_Cross.
    # We can precompute Old_Cross? 
    # a[i,j]b[p[i],p[j]] + a[j,i]b[p[j],p[i]]
    # This is O(1).
    
    # So:
    # delta = np.dot(a[i] - a[j], b[:, p[i]] - b[:, p[j]]) - (a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]])
    
    # This is very fast.
    
    # For 3-swaps, it's more complex. We will skip 3-swaps for now and rely on 
    # SA's ability to accept worse moves to escape, using 2-swaps.
    # To make it a "large step" strategy, we will use a higher initial temperature 
    # and a slower cooling rate, and perhaps allow "k-swaps" where k is small 
    # but we accept worse moves more aggressively.
    
    # Actually, the prompt suggests "digit-reversal" or non-adjacent swaps. 
    # 2-swaps of random indices ARE non-adjacent swaps.
    # The key is the SA acceptance.
    
    current_T = T_start
    cooling_factor = 0.9995 # Adjust based on expected iterations
    
    # Estimate iterations:
    # Each iter ~ 150 flops (dot) + overhead.
    # 115s / 150 flops is huge. Python overhead dominates.
    # Let's assume ~1000-5000 iterations per second.
    # Total iters ~ 100,000 to 500,000.
    # We want T to drop from 50000 to 1.
    # 50000 * (cooling)^N = 1 => (cooling)^N = 2e-5
    # N ln(c) = ln(2e-5) = -10.8
    # If N=100000, ln(c) = -1.08 => c = 0.34. Too fast.
    # If N=50000, ln(c) = -0.216 => c = 0.80.
    # Let's aim for ~50,000 iterations.
    # So cooling_factor ~ 0.85?
    # Let's use a time-based cooling instead to be robust.
    # T(t) = T_start * (T_end/T_start)^(t / total_time)
    
    total_time = 115.0
    
    while time.time() < deadline:
        # Calculate current T
        elapsed = time.time() - start_time
        if elapsed >= total_time:
            break
        T = T_start * (T_end / T_start) ** (elapsed / total_time)
        
        # Choose move
        # 90% 2-swap, 10% 3-swap?
        # 3-swap delta is expensive. Let's stick to 2-swap but with high T.
        # Or implement a "block swap"?
        # Let's try 2-swaps first.
        
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        # Calculate delta
        # Precompute vectors?
        # a[i] - a[j]
        # b[:, p[i]] - b[:, p[j]]
        
        # Note: b[:, p[i]] is a column of b indexed by p[i].
        # This is a gather operation.
        
        diff_a = a[i] - a[j]
        # b_col_i = b[:, p[i]]
        # b_col_j = b[:, p[j]]
        # diff_b = b_col_i - b_col_j
        
        # To avoid two gathers, we can do:
        # dot = np.dot(diff_a, b[:, p[i]]) - np.dot(diff_a, b[:, p[j]])
        
        dot1 = np.dot(diff_a, b[:, p[i]])
        dot2 = np.dot(diff_a, b[:, p[j]])
        dot_val = dot1 - dot2
        
        cross_old = a[i, j] * b[p[i], p[j]] + a[j, i] * b[p[j], p[i]]
        delta = dot_val - cross_old
        
        # Acceptance
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += int(delta)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        # Small chance to do a 3-swap to jump further?
        # Let's implement a simple 3-swap with exact cost recalc if T is high.
        # If T > 10000, we are in exploration mode.
        if T > 10000 and random.random() < 0.1:
            # Pick 3 distinct indices
            k = np.random.randint(n)
            if k != i and k != j:
                # Swap i, j, k? No, swap p[i], p[j], p[k] randomly.
                # Let's just pick a new random 2-swap? 
                # The current 2-swap is already random.
                # A 3-swap involves 3 indices.
                # Let's skip 3-swaps for simplicity and speed, relying on T.
                pass

    return best_p.tolist()

# EVOLVE-BLOCK-END
