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
    
    # 2. Initialize signs and sums
    rng = random.Random(42)
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
    
    sums = np.zeros(num_progs, dtype=np.int16)
    for p_idx, indices in enumerate(progressions):
        # Sum signs at these indices
        # Using numpy fancy indexing is fast
        s = np.sum(signs[list(indices)])
        sums[p_idx] = s
        
    current_max = np.max(np.abs(sums))
    best_max = current_max
    best_signs = signs.copy()
    
    t_start = time.time()
    iterations = 0
    
    # 3. Local Search Loop
    while time.time() - t_start < deadline - t0 - 2:
        iterations += 1
        
        # Strategy: Focus on the current worst progression
        # Find the progression with the max absolute sum
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
        affected_pids = index_to_progs[flip_idx]
        
        # Apply flip
        signs[flip_idx] = -old_sign
        sums[affected_pids] += delta
        
        # Calculate new max
        # Optimization: Only check if the max could have changed.
        # The max can only decrease if we reduced the absolute value of the current max.
        # The max can only increase if we increased the absolute value of some sums.
        # Simplest: just recalculate.
        new_max = np.max(np.abs(sums))
        
        if new_max < best_max:
            best_max = new_max
            best_signs = signs.copy()
            current_max = new_max
        elif new_max <= current_max:
            # Accept if it doesn't make things worse (plateau)
            current_max = new_max
            # Optional: if new_max == current_max, we might want to accept to move around
            # But strictly, we want to minimize.
            # Let's accept equal to allow wandering.
            pass
        else:
            # Revert
            signs[flip_idx] = old_sign
            sums[affected_pids] -= delta
            # current_max remains unchanged
            
        # Escape local minima: Random kick
        # Every 1000 iterations, try a random flip regardless of worst progression
        if iterations % 1000 == 0:
            flip_idx = rng.randint(0, n - 1)
            old_sign = signs[flip_idx]
            delta = -2 * old_sign
            affected_pids = index_to_progs[flip_idx]
            
            signs[flip_idx] = -old_sign
            sums[affected_pids] += delta
            
            new_max = np.max(np.abs(sums))
            
            if new_max < best_max:
                best_max = new_max
                best_signs = signs.copy()
                current_max = new_max
            elif new_max <= current_max + 1: # Allow slight increase to escape
                current_max = new_max
            else:
                signs[flip_idx] = old_sign
                sums[affected_pids] -= delta

    return [int(x) for x in best_signs]
