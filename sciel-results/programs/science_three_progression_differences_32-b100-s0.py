"""Program for maximizing 3-AP differences in a 32-element set within 1..512 using Simulated Annealing."""

import random
import time
import math

def count_aps(s):
    """
    Calculate the number of distinct differences d > 0 such that
    a, a+d, a+2d are all in the set s.
    """
    s_set = set(s)
    diffs = set()
    # Iterate over pairs (a, b) with a < b
    # d = b - a
    # Check if a + 2*d is in s_set
    # Note: We can optimize by iterating sorted elements
    elements = sorted(s)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = a + 2 * d
            if c in s_set:
                diffs.add(d)
    return len(diffs)

def construct():
    """
    Use Simulated Annealing to find a 32-element set in [1, 512]
    that maximizes the number of distinct 3-AP differences.
    """
    N = 32
    MAX_VAL = 512
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Initial candidate: A dense interval [1, 32] is a strong starting point (score ~15)
    # We will also try a few random starts and pick the best initial one
    initial_candidates = [
        list(range(1, N + 1)),
        random.sample(range(1, MAX_VAL + 1), N),
        sorted(random.sample(range(1, MAX_VAL // 2 + 1), N // 2) + random.sample(range(MAX_VAL // 2 + 1, MAX_VAL + 1), N - N // 2))
    ]
    
    best_init_score = -1
    current_set = None
    for cand in initial_candidates:
        score = count_aps(cand)
        if score > best_init_score:
            best_init_score = score
            current_set = cand
    
    current_score = best_init_score
    best_set = list(current_set)
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 10.0  # Initial temperature
    cooling_rate = 0.999  # Slight cooling per step
    min_T = 0.1
    
    # Precompute nothing, just use the set for lookups
    current_set_set = set(current_set)
    
    while time.time() - start_time < time_limit and T > min_T:
        # Perform a move: pick a random element and replace it with a random new value
        idx = random.randint(0, N - 1)
        old_val = current_set[idx]
        
        # Generate a new value
        # To avoid infinite loops picking existing values, we can just pick and check
        new_val = random.randint(1, MAX_VAL)
        
        # If new_val is already in the set, we need to handle it.
        # If new_val == old_val, skip.
        # If new_val != old_val and new_val in current_set_set, this move is invalid (duplicate).
        # We can retry or just reject. Rejecting is fine, it just wastes a bit of time.
        # To be more efficient, we can pick from the complement, but that's O(N).
        # Given MAX_VAL=512, collision probability is 32/512 = 6.25%. Acceptable.
        
        if new_val == old_val:
            continue
            
        if new_val in current_set_set:
            # Invalid move, effectively skip this step (or count as non-improving)
            # We'll just continue to next iteration to save time on evaluation
            continue
        
        # Apply move
        current_set_set.remove(old_val)
        current_set_set.add(new_val)
        current_set[idx] = new_val
        
        # Evaluate new score
        new_score = count_aps(current_set_set)
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set_set)
        else:
            # Revert
            current_set_set.remove(new_val)
            current_set_set.add(old_val)
            current_set[idx] = old_val
        
        # Cool down
        T *= cooling_rate
        
        # Optional: If we've been stuck for a while, reheat?
        # For now, simple cooling.
        
    # Final refinement: Local search (Hill Climbing) on the best found
    # This helps polish the solution
    if time.time() - start_time < time_limit - 5:
        current_set = list(best_set)
        current_set_set = set(current_set)
        current_score = best_score
        stuck_count = 0
        
        while time.time() - start_time < time_limit - 2 and stuck_count < 1000:
            improved = False
            elements = list(current_set_set)
            random.shuffle(elements)
            for old_val in elements:
                # Try a few random new values
                for _ in range(5):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val == old_val or new_val in current_set_set:
                        continue
                    
                    # Apply
                    current_set_set.remove(old_val)
                    current_set_set.add(new_val)
                    new_score = count_aps(current_set_set)
                    
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set_set)
                        improved = True
                        break # Restart loop with new set
                    else:
                        # Revert
                        current_set_set.remove(new_val)
                        current_set_set.add(old_val)
                
                if improved:
                    break
            
            if not improved:
                stuck_count += 1
                # If stuck, try a random perturbation
                if stuck_count % 10 == 0:
                    idx = random.randint(0, N - 1)
                    old_val = list(current_set_set)[idx]
                    new_val = random.randint(1, MAX_VAL)
                    if new_val != old_val and new_val not in current_set_set:
                        current_set_set.remove(old_val)
                        current_set_set.add(new_val)
                        current_score = count_aps(current_set_set)
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set_set)
                    stuck_count = 0 # Reset counter after perturbation
    
    return best_set
