import time
import numpy as np
from math import log

def score_set(U):
    """
    Compute the score for a set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    # Use vectorized operations for speed
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums and differences
    sums = np.add.outer(U_arr, U_arr)
    diffs = np.subtract.outer(U_arr, U_arr)
    
    # Count unique elements
    # Note: U-U includes negative values, 0, and positive values.
    # U+U includes only non-negative values.
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    
    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initial seeds: structured sets known to have good additive properties
    # We want large difference sets relative to sum sets.
    # Arithmetic progressions have small sum/diff ratio.
    # Geometric progressions have large sum/diff ratio.
    # Random sets are in between.
    
    seeds = [
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 9, 27, 81, 243, 729],
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        [0, 1, 4, 10, 20, 35, 56, 84, 120],
        [0, 1, 2, 4, 7, 13, 24, 44, 81],
    ]
    
    best_set = seeds[0]
    best_score = -1.0
    
    # Evaluate seeds
    for s in seeds:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = s
            
    # Simulated Annealing
    current_set = list(best_set)
    current_score = best_score
    
    # Parameters
    T_start = 0.5
    T_end = 0.01
    alpha = 0.995  # Cooling rate
    max_val_limit = 5000
    max_n_limit = 30
    
    T = T_start
    step = 0
    total_steps = 100000  # Arbitrary large, controlled by time
    
    while time.time() < deadline - 5:
        # Generate a random move
        move_type = np.random.rand()
        
        # 1. Add an element (50% probability if not too large)
        if move_type < 0.5 and len(current_set) < max_n_limit:
            # Choose a random value, biased towards existing range or slightly larger
            if current_set:
                max_val = max(current_set)
                # Sample from [1, max_val * 1.5] or [1, max_val_limit]
                upper = min(max_val * 2, max_val_limit)
                new_val = np.random.randint(1, upper + 1)
            else:
                new_val = 1
                
            if new_val not in current_set:
                new_set = current_set + [new_val]
                new_set.sort()
                new_score = score_set(new_set)
                
                # Acceptance criterion
                delta = new_score - current_score
                if delta > 0 or (T > 0 and np.random.rand() < np.exp(delta / T)):
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        
        # 2. Remove an element (30% probability if not too small)
        elif move_type < 0.8 and len(current_set) > 2:
            # Remove a random element (not 0)
            removable_indices = [i for i, x in enumerate(current_set) if x != 0]
            if removable_indices:
                idx = np.random.choice(removable_indices)
                new_set = [x for i, x in enumerate(current_set) if i != idx]
                new_score = score_set(new_set)
                
                delta = new_score - current_score
                if delta > 0 or (T > 0 and np.random.rand() < np.exp(delta / T)):
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        
        # 3. Swap an element (20% probability)
        else:
            if len(current_set) > 2:
                # Pick a random index to swap (not 0)
                swap_indices = [i for i, x in enumerate(current_set) if x != 0]
                if swap_indices:
                    idx = np.random.choice(swap_indices)
                    old_val = current_set[idx]
                    
                    # Generate new value
                    max_val = max(current_set)
                    upper = min(max_val * 2, max_val_limit)
                    new_val = np.random.randint(1, upper + 1)
                    
                    if new_val != old_val and new_val not in current_set:
                        new_set = current_set.copy()
                        new_set[idx] = new_val
                        new_set.sort()
                        new_score = score_set(new_set)
                        
                        delta = new_score - current_score
                        if delta > 0 or (T > 0 and np.random.rand() < np.exp(delta / T)):
                            current_set = new_set
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current_set)
        
        # Cool down
        T *= alpha
        step += 1
        
        # Restart if temperature is too low or stuck
        if T < T_end or step % 5000 == 0:
            # Occasionally restart from best found to escape local minima
            if step % 10000 == 0:
                current_set = list(best_set)
                current_score = best_score
            T = T_start
            step = 0
            
    # Final local polish using hill climbing on the best set found
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        max_iterations = 500
        iter_count = 0
        
        while improved and iter_count < max_iterations and time.time() < deadline - 1:
            improved = False
            iter_count += 1
            n = len(current)
            
            # Try replacing each element with nearby values
            for i in range(1, n):
                old_val = current[i]
                # Try a small range of deltas
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set.sort()
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
                    
            # Try adding an element
            if not improved and n < 25:
                max_val = max(current)
                for new_val in range(max_val - 5, max_val + 6):
                    if new_val <= 0 or new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                        
            # Try removing an element
            if not improved and n > 3:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                        
        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
