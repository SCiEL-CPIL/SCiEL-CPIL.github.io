import random
import time
import math

def calc_max_discrepancy(signs):
    n = len(signs)
    max_abs = 0
    # Pre-compute ranges? No, just efficient loop.
    # The bottleneck is the inner loop.
    for d in range(1, n):
        for r in range(d):
            curr_sum = 0
            # Unroll or just simple loop
            for idx in range(r, n, d):
                curr_sum += signs[idx]
                abs_sum = curr_sum if curr_sum > 0 else -curr_sum
                if abs_sum > max_abs:
                    max_abs = abs_sum
    return max_abs

def hill_climb(signs, limit_time):
    n = len(signs)
    current_max = calc_max_discrepancy(signs)
    improved = True
    while improved and time.time() < limit_time:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            if time.time() > limit_time:
                break
            old_sign = signs[i]
            signs[i] = -old_sign
            new_max = calc_max_discrepancy(signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                signs[i] = old_sign
    return signs, current_max

def construct():
    n = 256
    # Thue-Morse initialization
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    start_time = time.time()
    limit_time = start_time + 58.0 # 2s buffer
    
    # Initial Hill Climb
    best_signs, best_max = hill_climb(signs[:], limit_time)
    current_signs = best_signs[:]
    
    # Iterated Local Search
    while time.time() < limit_time:
        # Perturb the current solution
        # Number of flips: random between 2 and 5
        num_flips = random.randint(2, 5)
        perturbed = current_signs[:]
        for _ in range(num_flips):
            i = random.randint(0, n - 1)
            perturbed[i] = -perturbed[i]
        
        # Hill Climb from perturbed state
        # We allow this to run until it finds a local optimum or time runs out
        # But we must ensure we don't exceed the global time limit
        # If time is tight, we might skip or do a quick check
        new_signs, new_max = hill_climb(perturbed, limit_time)
        
        if new_max < best_max:
            best_max = new_max
            best_signs = new_signs
            current_signs = new_signs
        else:
            # Standard ILS: keep the new local optimum as the base for next perturbation
            # This allows wandering.
            # Alternatively, if we want to stay near the best, we could revert.
            # Let's try wandering (standard ILS).
            current_signs = new_signs
            
    return best_signs
