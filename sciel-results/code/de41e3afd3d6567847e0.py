import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    # Use 85 seconds to be safe against the 90s limit
    deadline = start_time + 85.0
    
    n = 30
    inv = [Fraction(1, i) for i in range(1, n + 1)]
    
    best_abs = Fraction(1, 1)
    best_coeffs = [0] * n
    
    def update_best(coeffs, val):
        nonlocal best_abs, best_coeffs
        if val != 0:
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_coeffs = coeffs[:]

    # List of initial configurations to try
    initial_configs = []
    
    # 1. Seed config
    seed_config = [0] * 28 + [1, -1]
    initial_configs.append(seed_config)
    
    # 2. Alternating signs
    initial_configs.append([1 if i % 2 == 0 else -1 for i in range(n)])
    initial_configs.append([-1 if i % 2 == 0 else 1 for i in range(n)])
    
    # 3. Random configs
    for _ in range(50):
        initial_configs.append([random.choice([-1, 0, 1]) for _ in range(n)])

    for init_coeffs in initial_configs:
        if time.time() > deadline:
            break
            
        coeffs = init_coeffs[:]
        current_sum = sum(c * inv[i] for i, c in enumerate(coeffs))
        
        if current_sum == 0:
            # Try to kick it out of zero
            for i in range(n):
                old_c = coeffs[i]
                new_c = 1 if old_c != 1 else -1
                delta = (new_c - old_c) * inv[i]
                trial_sum = current_sum + delta
                if trial_sum != 0:
                    coeffs[i] = new_c
                    current_sum = trial_sum
                    break
            if current_sum == 0:
                continue

        update_best(coeffs, current_sum)
        
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                old_c = coeffs[i]
                # Try flipping to other values
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    
                    delta = (new_c - old_c) * inv[i]
                    trial_sum = current_sum + delta
                    
                    if trial_sum == 0:
                        continue
                        
                    if abs(trial_sum) < abs(current_sum):
                        coeffs[i] = new_c
                        current_sum = trial_sum
                        improved = True
                        update_best(coeffs, current_sum)
                        break # Restart inner loop to pick up new best move

    # If we still have time, do more random restarts
    while time.time() < deadline:
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        current_sum = sum(c * inv[i] for i, c in enumerate(coeffs))
        
        if current_sum == 0:
            continue

        update_best(coeffs, current_sum)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                old_c = coeffs[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    
                    delta = (new_c - old_c) * inv[i]
                    trial_sum = current_sum + delta
                    
                    if trial_sum == 0:
                        continue
                        
                    if abs(trial_sum) < abs(current_sum):
                        coeffs[i] = new_c
                        current_sum = trial_sum
                        improved = True
                        update_best(coeffs, current_sum)
                        break

    return best_coeffs
