"""Rewritten program for low-discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill

    # Precompute the list of (start, step) pairs
    ap_pairs = []
    for step in range(1, n + 1):
        for start in range(n):
            ap_pairs.append((start, step))
    
    # Function to compute the maximum absolute signed sum on any AP
    # Also returns the (start, step) that achieves it
    def max_discrepancy(signs, return_argmax=False):
        max_disc = 0
        best_start = 0
        best_step = 1
        for start, step in ap_pairs:
            s = 0
            idx = start
            while idx < n:
                s += signs[idx]
                abs_s = abs(s)
                if abs_s > max_disc:
                    max_disc = abs_s
                    best_start = start
                    best_step = step
                idx += step
        if return_argmax:
            return max_disc, best_start, best_step
        return max_disc
    
    # Constraint-directed search:
    # 1. Find the AP(s) with max discrepancy
    # 2. Preferentially flip indices within those hot APs
    # 3. Use simulated annealing for escape from local minima
    
    rng = random.Random(123)
    
    # Multiple restarts with different initializations
    best_signs = None
    best_disc = float('inf')
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            signs = [1 if i % 2 == 0 else -1 for i in range(n)]
        elif restart == 1:
            signs = [1 if i % 3 < 1 else -1 for i in range(n)]
        elif restart == 2:
            signs = [1 if (i * 7 + 3) % 4 < 2 else -1 for i in range(n)]
        else:
            signs = [rng.choice([-1, 1]) for _ in range(n)]
        
        current_disc, cur_start, cur_step = max_discrepancy(signs, return_argmax=True)
        
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs[:]
        
        # Simulated annealing with constraint-directed moves
        temperature = 3.0
        cooling_rate = 0.9998
        min_temp = 0.05
        
        stall_counter = 0
        max_stall = 5000
        
        while time.time() - start_time < time_limit - 2.0:
            # With high probability, flip an index in the current worst AP
            if rng.random() < 0.7:
                # Pick an index from the worst AP
                ap_indices = list(range(cur_start, n, cur_step))
                if ap_indices:
                    idx = ap_indices[rng.randint(0, len(ap_indices) - 1)]
                else:
                    idx = rng.randint(0, n - 1)
            else:
                # Random flip to escape local minima
                idx = rng.randint(0, n - 1)
            
            signs[idx] = -signs[idx]
            
            new_disc, new_start, new_step = max_discrepancy(signs, return_argmax=True)
            
            delta = new_disc - current_disc
            
            if delta < 0:
                # Accept
                current_disc = new_disc
                cur_start = new_start
                cur_step = new_step
                stall_counter = 0
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
            else:
                # Accept with probability exp(-delta / temperature)
                if rng.random() < math.exp(-delta / temperature):
                    current_disc = new_disc
                    cur_start = new_start
                    cur_step = new_step
                else:
                    # Reject: flip back
                    signs[idx] = -signs[idx]
            
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp
            
            stall_counter += 1
            if stall_counter >= max_stall:
                break
    
    return best_signs

# EVOLVE-BLOCK-END
