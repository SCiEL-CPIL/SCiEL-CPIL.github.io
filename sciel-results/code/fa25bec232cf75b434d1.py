import numpy as np
import time
import random
import sys

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # 1. Generate all APs of length >= 2
    # We represent each AP by its start and step for fast indexing
    # But we need the actual indices to compute sums.
    # For n=256, the number of APs is manageable.
    ap_data = []  # list of (start, step, length, indices_array)
    for start in range(n):
        for step in range(1, n):
            # Max length is (n - 1 - start) // step + 1
            # We only care about APs with length >= 2
            # So we need at least 2 terms: start + step < n => step < n - start
            if step >= n - start:
                continue
            # Generate indices
            # Using np.arange is fast
            # start, start+step, ..., < n
            # length = (n - 1 - start) // step + 1
            # But we only include if length >= 2
            # Actually, the loop condition `if step >= n - start: continue` ensures at least 2 terms.
            # Let's verify: if start=0, step=1, terms: 0,1,...,255 (256 terms)
            # if start=254, step=1, terms: 254, 255 (2 terms)
            # if start=255, step=1, terms: 255 (1 term) -> skipped because step >= n-start (1 >= 1)
            # Wait, n-start for start=255 is 1. step=1. 1>=1 is true. So skipped. Correct.
            
            # Calculate length
            length = (n - 1 - start) // step + 1
            if length < 2:
                continue
            
            indices = np.arange(start, n, step, dtype=np.int32)
            ap_data.append((start, step, length, indices))
    
    num_aps = len(ap_data)
    
    # Precompute the list of APs for each position
    # pos_to_aps[pos] = list of ap_ids
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, (start, step, length, indices) in enumerate(ap_data):
        for i in indices:
            pos_to_aps[i].append(ap_id)

    # Convert pos_to_aps to numpy arrays for faster iteration?
    # Python lists of ints are okay for small sizes.
    # Average APs per position?
    # Total pairs = sum(lengths).
    # Let's just use lists.

    # 2. Helper: Compute AP sums efficiently
    # We maintain a numpy array `ap_sums` of shape (num_aps,)
    # Flipping sign at index i changes the sum of every AP containing i by 2*new_sign.
    
    def compute_initial_sums(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for ap_id, (start, step, length, indices) in enumerate(ap_data):
            sums[ap_id] = np.sum(signs[indices])
        return sums

    # 3. Helper: Get max absolute sum
    # np.max(np.abs(arr)) is fast for ~10k-50k elements.
    # Let's estimate num_aps.
    # For step=1: 255 APs
    # For step=2: ~128 APs
    # Sum roughly n log n? 
    # n=256.
    # Number of APs is bounded by n^2/2 = 32768.
    # Actually less.
    # Let's just use np.max.

    def get_max_abs_sum(ap_sums):
        return np.max(np.abs(ap_sums))

    # 4. Local Search: Hill Climbing with Random Restarts
    # We will use a simple bit-flip local search.
    # To escape local minima, we can use simulated annealing or random restarts.
    # Given the time limit, a few high-quality restarts with strong local search is good.
    
    def local_search(signs, max_iters, deadline, rng, temperature=0.0):
        signs = signs.copy()
        ap_sums = compute_initial_sums(signs)
        current_max = get_max_abs_sum(ap_sums)
        best_signs = signs.copy()
        best_max = current_max
        
        if temperature > 0:
            # Simulated Annealing
            # We need to track temperature
            pass
        
        no_improve = 0
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick random position
            i = rng.randint(0, n - 1)
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            affected = pos_to_aps[i]
            
            # Apply flip
            for ap_id in affected:
                ap_sums[ap_id] += delta
            
            new_max = get_max_abs_sum(ap_sums)
            
            if new_max < current_max:
                signs[i] = new_sign
                current_max = new_max
                no_improve = 0
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            elif new_max == current_max:
                # Accept sideways moves to wander
                if rng.random() < 0.5:
                    signs[i] = new_sign
                    # current_max unchanged
                    no_improve += 1
                else:
                    # Revert
                    for ap_id in affected:
                        ap_sums[ap_id] -= delta
                    no_improve += 1
            else:
                # Reject worsening moves (or accept with prob in SA)
                if temperature > 0:
                    delta_max = new_max - current_max
                    # Simple SA acceptance
                    if rng.random() < np.exp(-delta_max / temperature):
                        signs[i] = new_sign
                        current_max = new_max
                    else:
                        for ap_id in affected:
                            ap_sums[ap_id] -= delta
                else:
                    for ap_id in affected:
                        ap_sums[ap_id] -= delta
                no_improve += 1
                
            if no_improve > 200:
                # Restart local search from current state with higher temp?
                # Or just break to do a new restart?
                # Let's break and let the outer loop handle restarts.
                break

        return best_signs, best_max

    # 5. Main Loop
    best_overall_signs = None
    best_overall_max = n + 1
    
    # Strategy:
    # 1. Start with Thue-Morse (known good structure)
    # 2. Start with Random
    # 3. Use Local Search with varying intensities
    
    # Thue-Morse initialization
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
    
    # Run local search on Thue-Morse
    rng = random.Random(42)
    signs, max_val = local_search(thue_morse, 5000, deadline, rng)
    if max_val < best_overall_max:
        best_overall_max = max_val
        best_overall_signs = signs.copy()
    
    # Run local search on Random initializations
    # We have ~70 seconds left.
    # Each local search iteration involves:
    # - 1 randint
    # - loop over ~10-20 APs (avg affected)
    # - np.max over ~10k array
    # np.max is the bottleneck.
    # Let's estimate speed.
    
    # To maximize quality, we should do multiple restarts.
    # If we can do ~100 restarts, that's good.
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Alternate between random and perturbed best
        if r % 2 == 0:
            # Random init
            signs_init = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        else:
            # Perturb best
            signs_init = best_overall_signs.copy()
            for _ in range(10):
                idx = rng.randint(0, n - 1)
                signs_init[idx] *= -1
        
        # Use SA with cooling for some restarts to escape
        temp = 1.0 if r % 3 == 0 else 0.0
        signs, max_val = local_search(signs_init, 3000, deadline, rng, temperature=temp)
        
        if max_val < best_overall_max:
            best_overall_max = max_val
            best_overall_signs = signs.copy()
            # If we find a very good solution, spend more time polishing it
            if best_overall_max <= 4:
                # Intensive polish
                signs, max_val2 = local_search(best_overall_signs, 10000, deadline, rng)
                if max_val2 < best_overall_max:
                    best_overall_max = max_val2
                    best_overall_signs = signs.copy()

    # Final polish on the best solution found
    if best_overall_signs is not None:
        rng_final = random.Random(9999)
        signs, max_val = local_search(best_overall_signs, 5000, deadline, rng_final)
        if max_val < best_overall_max:
            best_overall_max = max_val
            best_overall_signs = signs.copy()

    if best_overall_signs is None:
        # Fallback
        rng = random.Random(0)
        return [rng.choice([-1, 1]) for _ in range(n)]
        
    return [int(s) for s in best_overall_signs]
