import time
import random
from math import lcm

def construct():
    # Precompute LCM and weights
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    W = [0] * 31
    for i in range(1, 31):
        W[i] = L // i
        
    best_abs_sum = float('inf')
    best_coeffs = None
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    def update_best(coeffs, sum_val):
        nonlocal best_abs_sum, best_coeffs
        if sum_val != 0 and abs(sum_val) < best_abs_sum:
            best_abs_sum = abs(sum_val)
            best_coeffs = coeffs[:]
            return best_abs_sum == 1
        return False

    # Strategy:
    # Indices 0-9 (i=1..10) are restricted to {-1, 1}
    # Indices 10-29 (i=11..30) are allowed {-1, 0, 1}
    
    while time.time() < deadline:
        c = [0] * 30
        # Init first 10 as +/- 1
        for i in range(10):
            c[i] = 1 if random.random() < 0.5 else -1
        # Init rest as -1, 0, 1
        for i in range(10, 30):
            c[i] = random.choice([-1, 0, 1])
            
        current_sum = 0
        for i in range(30):
            current_sum += c[i] * W[i+1]
            
        if update_best(c, current_sum):
            return best_coeffs
            
        improved = True
        stall_count = 0
        max_stall = 1000
        
        while improved and time.time() < deadline:
            improved = False
            for _ in range(10):
                idx = random.randint(0, 29)
                old_val = c[idx]
                
                if idx < 10:
                    # Flip
                    new_val = -old_val
                else:
                    # Change to another value in {-1, 0, 1}
                    choices = [v for v in [-1, 0, 1] if v != old_val]
                    new_val = random.choice(choices)
                    
                if new_val == old_val:
                    continue
                    
                delta = (new_val - old_val) * W[idx+1]
                new_sum = current_sum + delta
                
                if new_sum != 0 and abs(new_sum) < abs(current_sum):
                    current_sum = new_sum
                    c[idx] = new_val
                    improved = True
                    if update_best(c, current_sum):
                        return best_coeffs
                    break
                else:
                    stall_count += 1
                    if stall_count > max_stall:
                        improved = False
                        break
                        
    return best_coeffs
