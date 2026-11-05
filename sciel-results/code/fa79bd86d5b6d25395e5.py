import random
import time
import math

def calc_max_discrepancy(signs):
    n = len(signs)
    max_abs = 0
    # Convert to list of ints for fast access? It's already a list.
    # Optimization: Pre-calculate prefix sums? 
    # For an AP (start, step, len), sum is sum(signs[start::step][:len]).
    # We can just loop.
    for d in range(1, n):
        for r in range(d):
            curr_sum = 0
            for idx in range(r, n, d):
                curr_sum += signs[idx]
                abs_sum = curr_sum if curr_sum > 0 else -curr_sum
                if abs_sum > max_abs:
                    max_abs = abs_sum
    return max_abs

def simulated_annealing(signs, limit_time):
    n = len(signs)
    current_signs = signs[:]
    current_energy = calc_max_discrepancy(current_signs)
    best_signs = current_signs[:]
    best_energy = current_energy
    
    # Cooling parameters
    # We want to explore for a while, then settle.
    # T0 = 5.0, T_end = 0.01, alpha = 0.999? 
    # Number of steps is dynamic based on time.
    T = 5.0
    alpha = 0.9995 # Cooling factor
    T_min = 0.01
    
    start_time = time.time()
    
    while time.time() < limit_time:
        if T < T_min:
            break
            
        # Pick a random index to flip
        i = random.randint(0, n - 1)
        old_sign = current_signs[i]
        current_signs[i] = -old_sign
        
        new_energy = calc_max_discrepancy(current_signs)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_signs = current_signs[:]
        else:
            current_signs[i] = old_sign
            
        T *= alpha
        
    return best_signs, best_energy

def construct():
    n = 256
    # Initialize with Thue-Morse as it's a good starting point for low discrepancy
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    start_time = time.time()
    limit_time = start_time + 58.0
    
    # Run Simulated Annealing
    best_signs, best_max = simulated_annealing(signs, limit_time)
    
    # Optional: Do a final hill climb on the best found to polish
    # Since SA might stop at a local min that isn't the global one, 
    # but a quick hill climb on the result is cheap and safe.
    # We implement a simple hill climb for the remaining time.
    current_signs = best_signs[:]
    current_max = best_max
    improved = True
    while improved and time.time() < limit_time:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            if time.time() > limit_time:
                break
            old_sign = current_signs[i]
            current_signs[i] = -old_sign
            new_max = calc_max_discrepancy(current_signs)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                current_signs[i] = old_sign
                
    if current_max < best_max:
        best_signs = current_signs
        best_max = current_max
        
    return best_signs
