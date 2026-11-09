import time
import random
import math
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=10 and n=20
    # For n=10: 55 pairs
    # For n=20: 210 pairs
    pairs_10_i = np.array([i for i in range(10) for j in range(i, 10)], dtype=np.int32)
    pairs_10_j = np.array([j for i in range(10) for j in range(i, 10)], dtype=np.int32)
    
    pairs_20_i = np.array([i for i in range(20) for j in range(i, 20)], dtype=np.int32)
    pairs_20_j = np.array([j for i in range(20) for j in range(i, 20)], dtype=np.int32)

    def evaluate_10(A):
        """Compute max(|A+A|, |A*A|) for set A of size 10."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_10_i] + A_arr[pairs_10_j]
        prods = A_arr[pairs_10_i] * A_arr[pairs_10_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)

    def evaluate_20(A):
        """Compute max(|A+A|, |A*A|) for set A of size 20."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_20_i] + A_arr[pairs_20_j]
        prods = A_arr[pairs_20_i] * A_arr[pairs_20_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)

    # Generate the pool of 5-smooth numbers (Hamming numbers) <= max_val
    # Form: 2^a * 3^b * 5^c
    smooth_pool = set()
    for a in range(0, 14):  # 2^13 = 8192
        v2 = 1 << a
        if v2 > max_val: break
        for b in range(0, 9): # 3^8 = 6561
            v3 = v2 * (3 ** b)
            if v3 > max_val: break
            for c in range(0, 6): # 5^5 = 3125
                v5 = v3 * (5 ** c)
                if v5 > max_val: break
                smooth_pool.add(v5)
    
    smooth_list = sorted(list(smooth_pool))
    num_smooth = len(smooth_list)
    
    # Map values to indices for quick lookup
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # --- Phase 1: Generate a library of high-quality 10-element subsets ---
    # We want subsets that are internally compact (small sum/product sets).
    # We will generate several candidates and keep the best ones.
    
    library = []
    num_blocks_to_find = 50 # Target number of good blocks
    
    # Strategy for generating blocks:
    # 1. Start with a random smooth number
    # 2. Greedily add numbers that minimize the increase in internal complexity
    # 3. Perform local search (swaps) to refine
    
    def generate_block():
        # Random start
        current_indices = [random.randint(0, num_smooth - 1)]
        current_vals = [smooth_list[current_indices[0]]]
        
        # Greedy fill
        for _ in range(9):
            best_idx = -1
            best_score = float('inf')
            # Try all unused smooth numbers
            for idx in range(num_smooth):
                if idx in current_indices:
                    continue
                new_indices = current_indices + [idx]
                new_vals = current_vals + [smooth_list[idx]]
                # Quick proxy: just check the new element's interaction? 
                # No, evaluate_10 is fast enough for 10 elements (55 pairs)
                score = evaluate_10(new_vals)
                if score < best_score:
                    best_score = score
                    best_idx = idx
            current_indices.append(best_idx)
            current_vals.append(smooth_list[best_idx])
            
        # Local Search on the block
        current_score = evaluate_10(current_vals)
        improved = True
        # Time limit per block: ~1s
        block_deadline = time.time() + 1.0
        
        while improved and time.time() < block_deadline:
            improved = False
            # Try swapping each element
            for i in range(10):
                old_idx = current_indices[i]
                old_val = current_vals[i]
                best_swap_idx = -1
                best_swap_score = current_score
                
                # Sample a subset of candidates to save time if pool is large
                # But pool is ~500, so full scan is okay (500 * 55 ops is small)
                for new_idx in range(num_smooth):
                    if new_idx == old_idx or new_idx in current_indices:
                        continue
                    current_indices[i] = new_idx
                    current_vals[i] = smooth_list[new_idx]
                    s = evaluate_10(current_vals)
                    if s < best_swap_score:
                        best_swap_score = s
                        best_swap_idx = new_idx
                
                if best_swap_idx != -1:
                    current_indices[i] = best_swap_idx
                    current_vals[i] = smooth_list[best_swap_idx]
                    current_score = best_swap_score
                    improved = True
                else:
                    # Revert if no improvement found for this i (though we didn't change it unless found)
                    # Actually, the loop above only updates if found. If not found, indices[i] is still old_idx?
                    # Wait, in the inner loop I modify current_indices[i]. I need to revert if no best_swap.
                    current_indices[i] = old_idx
                    current_vals[i] = old_val
                    
        return tuple(sorted(current_vals)), current_score

    # Generate library
    # We run this in parallel-ish by just looping, but keeping it fast.
    # 50 blocks * 1s = 50s. Too slow.
    # Let's reduce time per block to 0.5s and reduce number of blocks to 20.
    # Or make the greedy step faster.
    
    # Optimized generation:
    # 1. Greedy step is O(N^2 * Eval). N=500, Eval=55 ops. 500*55*10 = 275k ops. Fast.
    # 2. Local search is the bottleneck.
    
    # Let's try 20 blocks, 0.5s each. Total 10s.
    num_blocks = 20
    block_time_limit = 0.5
    
    for _ in range(num_blocks):
        if time.time() > deadline - 20: # Reserve 20s for Phase 2
            break
        block, score = generate_block()
        # Deduplicate
        if block not in [b[0] for b in library]:
            library.append((block, score))
            
    # Sort library by score (lower is better)
    library.sort(key=lambda x: x[1])
    
    # --- Phase 2: Combine blocks to form 20-element sets ---
    # We pick two blocks (A1, A2) such that A1 and A2 are disjoint.
    # The combined set is A = A1 U A2.
    # We want to minimize evaluate_20(A).
    
    best_set = None
    best_score = float('inf')
    
    # Precompute sets for fast intersection check
    library_sets = [set(b[0]) for b in library]
    
    # Iterate over pairs of blocks
    # To save time, we only consider the top K blocks for pairing.
    K = min(len(library), 15)
    
    for i in range(K):
        for j in range(i + 1, K):
            if time.time() > deadline - 5:
                return sorted(best_set) if best_set else sorted(smooth_list[:20])
            
            if library_sets[i].intersection(library_sets[j]):
                continue
            
            combined_vals = list(library_sets[i].union(library_sets[j]))
            if len(combined_vals) != 20:
                continue
                
            score = evaluate_20(combined_vals)
            if score < best_score:
                best_score = score
                best_set = combined_vals[:]
                
    # --- Phase 3: Local Search on the best combined set ---
    # Refine the best set found in Phase 2.
    if best_set:
        current_vals = best_set[:]
        current_score = best_score
        
        # Use the full smooth pool for swaps
        # We perform hill climbing with random restarts if stuck
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try swapping each element
            for i in range(20):
                old_val = current_vals[i]
                best_swap_val = -1
                best_swap_score = current_score
                
                # Sample candidates to speed up? 
                # Full scan of 500 candidates * 20 elements * Eval(210 pairs)
                # 10,000 evals per pass. 210 * 10000 = 2.1M ops. Fast in numpy.
                
                for new_val in smooth_list:
                    if new_val == old_val or new_val in current_vals:
                        continue
                    
                    current_vals[i] = new_val
                    s = evaluate_20(current_vals)
                    if s < best_swap_score:
                        best_swap_score = s
                        best_swap_val = new_val
                
                if best_swap_val != -1:
                    current_vals[i] = best_swap_val
                    current_score = best_swap_score
                    improved = True
                else:
                    current_vals[i] = old_val
            
            # If stuck, kick
            if not improved and time.time() < deadline - 5:
                # Randomly change 1-2 elements
                for _ in range(2):
                    idx = random.randint(0, 19)
                    old_v = current_vals[idx]
                    new_v = random.choice(smooth_list)
                    while new_v in current_vals:
                        new_v = random.choice(smooth_list)
                    current_vals[idx] = new_v
                current_score = evaluate_20(current_vals)
                # Note: This might worsen the score, but the next hill climb pass will try to fix it.
                # To be safe, we only accept kicks if they don't increase score by too much?
                # Or just rely on the hill climb to recover.
                # Given the time budget, a simple kick is fine.

        if current_score < best_score:
            best_score = current_score
            best_set = current_vals[:]

    if best_set is None:
        # Fallback: just take the first 20 smooth numbers
        best_set = smooth_list[:20]

    return sorted(best_set)
