import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55  # 5s safety margin
    
    # 1. Precompute all arithmetic progressions
    # A progression is defined by start 'a', step 'd', length 'k'
    # Indices: a, a+d, ..., a+(k-1)d
    # Constraint: a + (k-1)d < n
    # We store each progression as a tuple of indices.
    # We also build a reverse map: index -> list of progression IDs
    
    progressions = []
    index_to_progs = [[] for _ in range(n)]
    
    pid = 0
    for d in range(1, n):
        for a in range(d):
            # Max length k such that a + (k-1)d < n
            # k-1 < (n - a) / d
            # k < (n - a) / d + 1
            max_k = (n - 1 - a) // d + 1
            for k in range(1, max_k + 1):
                indices = tuple(a + j * d for j in range(k))
                progressions.append(indices)
                for idx in indices:
                    index_to_progs[idx].append(pid)
                pid += 1
                
    num_progs = len(progressions)
    
    # 2. Initialize signs using Thue-Morse sequence
    # Thue-Morse is known to have low discrepancy properties.
    # t_i = popcount(i) % 2. Map 0 -> 1, 1 -> -1 (or vice versa).
    signs = np.array([1 if (bin(i).count('1') % 2 == 0) else -1 for i in range(n)], dtype=np.int8)
    
    # Precompute sums for all progressions
    sums = np.zeros(num_progs, dtype=np.int16)
    for p_idx, indices in enumerate(progressions):
        s = np.sum(signs[list(indices)])
        sums[p_idx] = s
        
    current_max = np.max(np.abs(sums))
    best_max = current_max
    best_signs = signs.copy()
    
    # Precompute affected indices as numpy arrays for faster updates
    # affected_pids[i] is the list of progression IDs that include index i
    # We convert these to numpy arrays for vectorized addition
    affected_pids_arrays = [np.array(lst, dtype=np.int32) for lst in index_to_progs]
    
    rng = random.Random(42)
    t_start = time.time()
    iterations = 0
    
    # 3. Local Search Loop with Simulated Annealing-like acceptance
    # We use a simple heuristic: accept if better, or with some probability if worse (to escape local minima)
    # But since we are minimizing the MAX, a small increase might be tolerable if it allows a large decrease later.
    # However, standard hill climbing is often robust. Let's try a variant:
    # - Always accept if new_max < current_max
    # - Accept with probability p if new_max == current_max (to wander)
    # - Reject if new_max > current_max (strict hill climbing) OR use a small temperature?
    # Given the time constraint, strict hill climbing with random restarts/kicks is safer.
    
    # Let's implement a "kick" strategy: if stuck for X iterations, do a random flip that might worsen the score
    # to escape the local minimum.
    
    stuck_count = 0
    max_stuck = 500
    
    while time.time() - t_start < deadline - t0 - 2:
        iterations += 1
        
        # Strategy: Focus on the current worst progression
        worst_pid = np.argmax(np.abs(sums))
        worst_indices = progressions[worst_pid]
        
        # Try flipping a random index from the worst progression
        if len(worst_indices) > 0:
            flip_idx = rng.choice(worst_indices)
        else:
            flip_idx = rng.randint(0, n - 1)
            
        old_sign = signs[flip_idx]
        delta = -2 * old_sign
        
        # Identify affected progressions
        affected_pids = affected_pids_arrays[flip_idx]
        
        # Apply flip
        signs[flip_idx] = -old_sign
        sums[affected_pids] += delta
        
        # Calculate new max
        new_max = np.max(np.abs(sums))
        
        if new_max < best_max:
            best_max = new_max
            best_signs = signs.copy()
            current_max = new_max
            stuck_count = 0
        elif new_max <= current_max:
            # Accept if it doesn't make things worse
            current_max = new_max
            stuck_count = 0
        else:
            # Revert
            signs[flip_idx] = old_sign
            sums[affected_pids] -= delta
            stuck_count += 1
            
            # Escape local minima: Random kick
            if stuck_count > max_stuck:
                stuck_count = 0
                # Pick a few random indices to flip, accepting any change
                for _ in range(3):
                    kick_idx = rng.randint(0, n - 1)
                    kick_old_sign = signs[kick_idx]
                    kick_delta = -2 * kick_old_sign
                    kick_affected = affected_pids_arrays[kick_idx]
                    
                    signs[kick_idx] = -kick_old_sign
                    sums[kick_affected] += kick_delta
                    
                    # Don't check for improvement here, just move
                    # Update current_max to reflect the new state
                    current_max = np.max(np.abs(sums))
                    
                    # If this kick happened to improve best, update
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()

    return [int(x) for x in best_signs]
