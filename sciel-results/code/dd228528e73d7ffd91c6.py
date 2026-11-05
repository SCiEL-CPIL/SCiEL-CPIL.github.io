"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute LCM and integer weights for exact arithmetic
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // i for i in range(1, 31)]  # weights[i] corresponds to 1/(i+1)
    
    best_nonzero_abs = None
    best_nonzero_coeffs = None
    
    def calculate_N(coeffs):
        return sum(c * w for c, w in zip(coeffs, weights))
    
    def update_best(coeffs, N):
        nonlocal best_nonzero_abs, best_nonzero_coeffs
        if N == 0:
            return
        abs_N = abs(N)
        if best_nonzero_abs is None or abs_N < best_nonzero_abs:
            best_nonzero_abs = abs_N
            best_nonzero_coeffs = list(coeffs)
    
    def random_diff_val(current):
        options = [-1, 0, 1]
        options.remove(current)
        return random.choice(options)
    
    def local_search(coeffs, N, time_limit):
        """Steepest-descent local search minimizing |N|, avoiding 0."""
        nonlocal best_nonzero_abs, best_nonzero_coeffs
        
        if N == 0:
            # Find nearest nonzero
            best_move = None
            best_abs = None
            for i in range(30):
                for new_val in (-1, 0, 1):
                    if new_val == coeffs[i]:
                        continue
                    delta = (new_val - coeffs[i]) * weights[i]
                    new_N = N + delta
                    if new_N != 0:
                        a = abs(new_N)
                        if best_abs is None or a < best_abs:
                            best_abs = a
                            best_move = (i, new_val)
            if best_move:
                i, new_val = best_move
                coeffs[i] = new_val
                N = calculate_N(coeffs)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_move = None
            best_abs = None
            current_abs = abs(N)
            
            for i in range(30):
                for new_val in (-1, 0, 1):
                    if new_val == coeffs[i]:
                        continue
                    delta = (new_val - coeffs[i]) * weights[i]
                    new_N = N + delta
                    if new_N == 0:
                        continue
                    a = abs(new_N)
                    if a < current_abs:
                        if best_abs is None or a < best_abs:
                            best_abs = a
                            best_move = (i, new_val, new_N)
            
            if best_move:
                i, new_val, new_N = best_move
                coeffs[i] = new_val
                N = new_N
                improved = True
                update_best(coeffs, N)
        
        update_best(coeffs, N)
        return coeffs, N
    
    def simulated_annealing(coeffs, N, time_limit):
        """Simulated annealing to escape local minima."""
        nonlocal best_nonzero_abs, best_nonzero_coeffs
        
        if N == 0:
            for i in range(30):
                for new_val in (-1, 0, 1):
                    if new_val == coeffs[i]:
                        continue
                    delta = (new_val - coeffs[i]) * weights[i]
                    new_N = N + delta
                    if new_N != 0:
                        coeffs[i] = new_val
                        N = new_N
                        break
                if N != 0:
                    break
        
        update_best(coeffs, N)
        
        T_start = max(weights) / 5.0
        T = T_start
        alpha = 0.9995
        
        while time.time() < time_limit:
            i = random.randint(0, 29)
            old_val = coeffs[i]
            new_val = random_diff_val(old_val)
            
            delta_N = (new_val - old_val) * weights[i]
            new_N = N + delta_N
            
            curr_cost = abs(N)
            new_cost = abs(new_N)
            delta_cost = new_cost - curr_cost
            
            if delta_cost < 0:
                coeffs[i] = new_val
                N = new_N
                update_best(coeffs, N)
            elif new_N == 0:
                # Avoid staying at zero; accept with high probability to explore
                if T > 1e-6:
                    if random.random() < 0.5:
                        coeffs[i] = new_val
                        N = new_N
            else:
                if T > 1e-6:
                    ratio = delta_cost / T
                    if ratio < 50:
                        p = math.exp(-ratio)
                        if random.random() < p:
                            coeffs[i] = new_val
                            N = new_N
                            update_best(coeffs, N)
            
            T *= alpha
        
        # Finish with local search
        coeffs, N = local_search(coeffs, N, time_limit)
        return coeffs, N
    
    random.seed(42)
    
    # Generate diverse starting points
    start_configs = []
    
    # 1. Random dense starts
    for _ in range(50):
        coeffs = []
        for i in range(30):
            r = random.random()
            if r < 0.35:
                coeffs.append(1)
            elif r < 0.70:
                coeffs.append(-1)
            else:
                coeffs.append(0)
        start_configs.append(coeffs)
    
    # 2. Structured starts
    start_configs.append([1] * 30)
    start_configs.append([-1] * 30)
    start_configs.append([1 if i % 2 == 0 else -1 for i in range(30)])
    start_configs.append([1 if i % 2 == 0 else 0 for i in range(30)])
    start_configs.append([0] * 30)
    start_configs[29] = 1
    start_configs.append([0] * 29 + [1])
    start_configs.append([1] + [0] * 29)
    start_configs.append([-1] + [0] * 29)
    
    # 3. Split starts (first half +1, second half -1)
    for split in range(5, 26, 3):
        coeffs = [1] * split + [-1] * (30 - split)
        start_configs.append(coeffs)
    
    # Process each start: local search first, then SA if time remains
    for idx, coeffs in enumerate(start_configs):
        if time.time() > deadline:
            break
        
        N = calculate_N(coeffs)
        
        # First do local search with a short time budget
        ls_time = min(1.0, (deadline - time.time()) * 0.3)
        if ls_time < 0.1:
            ls_time = 0.1
        coeffs, N = local_search(coeffs, N, time.time() + ls_time)
        
        # If time remains, do simulated annealing
        remaining = deadline - time.time()
        if remaining > 2.0:
            sa_time = min(remaining * 0.5, 5.0)
            if sa_time > 0.5:
                coeffs, N = simulated_annealing(coeffs, N, time.time() + sa_time)
    
    if best_nonzero_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_nonzero_coeffs
# EVOLVE-BLOCK-END
