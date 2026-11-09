import time
import random
import math
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)
    
    # Generate the pool of 7-smooth numbers <= max_val
    # 7-smooth numbers are of the form 2^a * 3^b * 5^c * 7^d
    smooth_pool = set()
    for a in range(0, 14):  # 2^13 = 8192, 2^14 > 10000
        v2 = 1 << a
        if v2 > max_val: break
        for b in range(0, 9): # 3^8 = 6561
            v3 = v2 * (3 ** b)
            if v3 > max_val: break
            for c in range(0, 6): # 5^5 = 3125
                v5 = v3 * (5 ** c)
                if v5 > max_val: break
                for d in range(0, 5): # 7^4 = 2401
                    v7 = v5 * (7 ** d)
                    if v7 > max_val: break
                    smooth_pool.add(v7)
    
    smooth_list = sorted(list(smooth_pool))
    # Filter to a manageable size if the pool is too large, though 7-smooth <= 10000 is ~100-200
    # Let's keep all of them for now, it's small enough.
    if len(smooth_list) > 300:
        # Bias towards smaller numbers which are denser and create smaller sums/products
        smooth_list = sorted(smooth_list)[:300]
    
    # Mapping from value to index in smooth_list for fast lookup
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Initial set: Pick the first n smooth numbers (smallest ones)
    # This minimizes both sum and product sets initially
    initial_A = smooth_list[:n]
    current_score = evaluate(initial_A)
    best_set = initial_A[:]
    best_score = current_score
    
    # Local Search: Hill Climbing on the restricted smooth number space
    # We represent the set as a list of indices into smooth_list
    current_indices = [val_to_idx[v] for v in initial_A]
    current_set_vals = initial_A[:]
    
    # To speed up evaluation, we can cache sums and products? 
    # No, numpy is fast enough. But we can precompute pair sums/prods for the smooth pool?
    # The pool is ~150 items. 150*150 = 22500 pairs.
    # We are picking 20. 
    # Let's stick to numpy evaluation on the 20x20 subset.
    
    improved = True
    max_iters = 0
    while improved and time.time() < deadline - 5:
        improved = False
        # Try to improve each element
        for i in range(n):
            old_idx = current_indices[i]
            old_val = smooth_list[old_idx]
            
            # Try swapping with other smooth numbers
            # Heuristic: Try nearby indices first, then random
            # We can't just try all 300 for each of 20 elements every iteration (6000 evals * 50000 ops = 300M ops)
            # 300M ops in numpy might be ~0.3-1s. So 1-3 iterations per second.
            # We have 80s. So ~80-240 iterations.
            # Each iteration tries 20 * 300 = 6000 swaps.
            # Total evals ~ 500k - 1.5M.
            # This is feasible.
            
            # Shuffle candidates to avoid bias, but keep small ones prioritized?
            # Actually, let's just iterate through all candidates in the pool.
            # To make it faster, we can randomize the order of candidates or elements.
            
            for new_idx in range(len(smooth_list)):
                if new_idx == old_idx:
                    continue
                
                new_val = smooth_list[new_idx]
                
                # Check distinctness
                # Since we are iterating indices, we need to ensure new_val is not in the set
                # We maintain a set of values for O(1) lookup
                # But modifying the set in the loop is tricky if we revert.
                # Let's just check against current_set_vals
                # current_set_vals is a list of 20. O(20) check is fine.
                
                if new_val in current_set_vals:
                    continue
                
                # Swap
                current_set_vals[i] = new_val
                current_indices[i] = new_idx
                
                new_score = evaluate(current_set_vals)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current_set_vals[:]
                else:
                    # Revert
                    current_set_vals[i] = old_val
                    current_indices[i] = old_idx
                    
        # If we didn't improve in a full pass, we might be stuck in a local minimum.
        # Try a random restart or perturbation?
        # Given the restricted space, let's try a few random restarts from the best found so far
        # with a "kick" (randomly change 1-2 elements to random smooth numbers)
        
        if not improved:
             # Perturbation
             if time.time() < deadline - 10:
                 # Kick
                 k = random.randint(1, 3)
                 for _ in range(k):
                     idx_to_change = random.randint(0, n-1)
                     new_idx = random.randint(0, len(smooth_list)-1)
                     new_val = smooth_list[new_idx]
                     if new_val not in current_set_vals:
                         current_set_vals[idx_to_change] = new_val
                         current_indices[idx_to_change] = new_idx
                         current_score = evaluate(current_set_vals)
                         # Don't update best here, let the hill climbing do it
                         # But if this kick lands on a better spot, hill climbing will find it.
                         # Actually, if the kick makes it worse, hill climbing might get stuck again.
                         # Let's just let the main loop handle it. If it doesn't improve, we loop again.
                         # But to avoid infinite loops on bad kicks, we should limit kicks.
                         # The `improved` flag will be False again, leading to another kick.
                         # This is effectively simulated annealing with T=0 (greedy) plus random restarts.
    
    return sorted(best_set)
