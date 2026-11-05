import random
import time
from math import lcm, exp

def construct():
    deadline = time.time() + 88
    N = 30
    
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
        
    weights = [L // i for i in range(1, N + 1)]
    
    # Seed: 1/29 - 1/30
    best_coeffs = [0] * N
    best_coeffs[28] = 1
    best_coeffs[29] = -1
    best_sum = weights[28] - weights[29]
    best_abs = abs(best_sum)
    
    if best_abs == 1:
        return best_coeffs
        
    # Simulated Annealing
    T_start = float(L)
    T_end = 1.0
    # Decay factor per step
    # We estimate ~10M steps in 60s? Maybe less. 
    # Let's use time-based cooling or a large number of steps.
    # 1 step is ~100ns in C, ~1us in Python. 
    # 10M steps * 1us = 10s. 
    # Let's target 50M steps.
    # Decay: T = T_start * (T_end/T_start)^(step / total_steps)
    # Or simpler: T *= 0.99999
    
    current_coeffs = best_coeffs[:]
    current_sum = best_sum
    current_abs = best_abs
    
    T = T_start
    decay = 0.99999
    
    stuck_count = 0
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Move selection
        # 1-flip
        k1 = random.randint(0, N - 1)
        old1 = current_coeffs[k1]
        # Pick a different value
        # Values: -1, 0, 1
        # If old is 0, new is -1 or 1.
        # If old is 1, new is 0 or -1.
        # If old is -1, new is 0 or 1.
        if old1 == 0:
            new1 = random.choice([-1, 1])
        elif old1 == 1:
            new1 = random.choice([0, -1])
        else:
            new1 = random.choice([0, 1])
            
        delta = (new1 - old1) * weights[k1]
        new_sum = current_sum + delta
        
        accept = False
        if new_sum == 0:
            if current_sum == 0:
                accept = True
            else:
                accept = False
        else:
            new_abs = abs(new_sum)
            if new_abs < current_abs:
                accept = True
            else:
                delta_e = new_abs - current_abs
                if delta_e > 0:
                    # Accept if random < exp(-delta_e / T)
                    # If delta_e / T is large, exp is ~0
                    ratio = delta_e / T
                    if ratio > 50:
                        accept = False
                    else:
                        accept = random.random() < exp(-ratio)
                else:
                    accept = True
                    
        if accept:
            current_coeffs[k1] = new1
            current_sum = new_sum
            current_abs = abs(new_sum)
            
            if current_abs < best_abs:
                best_abs = current_abs
                best_coeffs = current_coeffs[:]
                stuck_count = 0
                if best_abs == 1:
                    return best_coeffs
            else:
                stuck_count += 1
        else:
            stuck_count += 1
            
        # Random restart if stuck
        if stuck_count > 5000:
            # Revert to best
            current_coeffs = best_coeffs[:]
            current_sum = best_sum
            current_abs = best_abs
            stuck_count = 0
            # Optional: Random kick to escape basin
            # for _ in range(3):
            #    k = random.randint(0, N-1)
            #    current_coeffs[k] = random.choice([-1, 0, 1])
            #    # Recalc sum? No, just restart from best.
            
        T *= decay
        if T < T_end:
            T = T_end
            
    return best_coeffs
