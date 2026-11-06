import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    best_A = None
    best_max_count = float('inf')
    
    # Target interval
    target_start = 128
    target_end = 256
    num_targets = target_end - target_start + 1
    target_range = range(target_start, target_end + 1)
    
    # Helper to calculate max count and check validity
    def evaluate(A_set):
        counts = [0] * 257
        A_list = list(A_set)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            for j in range(i, n):
                b = A_list[j]
                s = a + b
                if 0 <= s <= 256:
                    counts[s] += 1
        
        max_c = 0
        missing = 0
        sum_sq = 0
        for s in target_range:
            c = counts[s]
            if c == 0:
                missing += 1
            if c > max_c:
                max_c = c
            sum_sq += c * c
        
        # If missing, it's invalid
        if missing > 0:
            return None, None, None
        
        return max_c, sum_sq, counts

    # Local Search
    def local_search(A_init, max_iters=50000):
        A_set = set(A_init)
        A_list = list(A_set)
        
        # Initial evaluation
        max_c, sum_sq, counts = evaluate(A_set)
        if max_c is None:
            return None, float('inf')
        
        current_max = max_c
        current_sum_sq = sum_sq
        best_local_max = current_max
        best_local_set = set(A_set)
        
        # Precompute potential moves? No, dynamic.
        # We want to minimize sum_sq to flatten, which should help max.
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random element to remove
            if not A_list:
                break
            idx_remove = random.randint(0, len(A_list) - 1)
            x = A_list[idx_remove]
            
            # Pick a random element to add
            # To speed up, pick from a pool of candidates or just random from [0, 128]
            y = random.randint(0, 128)
            if y == x or y in A_set:
                continue
                
            # Apply move
            A_set.remove(x)
            A_set.add(y)
            A_list[idx_remove] = y
            
            # Evaluate new state
            new_max, new_sum_sq, new_counts = evaluate(A_set)
            
            if new_max is None:
                # Invalid, rollback
                A_set.remove(y)
                A_set.add(x)
                A_list[idx_remove] = x
                continue
            
            # Acceptance criteria
            # We prioritize reducing max. If max is same, we accept if sum_sq decreases.
            if new_max < current_max:
                current_max = new_max
                current_sum_sq = new_sum_sq
                if current_max < best_local_max:
                    best_local_max = current_max
                    best_local_set = set(A_set)
            elif new_max == current_max and new_sum_sq < current_sum_sq:
                current_max = new_max
                current_sum_sq = new_sum_sq
                # Update best if sum_sq is better? No, best is tracked by max.
                # But a flatter distribution might allow further max reduction later.
                # So we keep the state.
            else:
                # Rollback
                A_set.remove(y)
                A_set.add(x)
                A_list[idx_remove] = x
                
        return best_local_set, best_local_max

    # Try different sizes and random seeds
    for k in range(30, 50):
        if time.time() > deadline:
            break
            
        # Multiple restarts for this size
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Generate initial set
            # Random subset
            A_init = random.sample(range(129), k)
            
            # Run local search
            res_set, res_max = local_search(A_init)
            
            if res_set is not None and res_max < best_max_count:
                best_max_count = res_max
                best_A = res_set
                
    # If no valid set found (should not happen), return seed
    if best_A is None:
        return list(range(64, 129))
        
    return list(best_A)

# EVOLVE-BLOCK-END
