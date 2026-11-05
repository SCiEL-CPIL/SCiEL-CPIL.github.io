import time
import random
import math
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # 90s limit, keep 5s margin
    
    # Precompute LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // i for i in range(1, 31)]
    
    best_cost = L  # Max possible cost
    best_coeffs = None
    
    num_runs = 50
    time_per_run = (deadline - time.time()) / num_runs
    
    random.seed(42)
    
    for run in range(num_runs):
        if time.time() > deadline - 2:
            break
            
        # Initialize
        # Start with random -1, 0, 1
        coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        current_sum = 0
        for i in range(30):
            current_sum += coeffs[i] * weights[i]
            
        if current_sum == 0:
            current_cost = L
        else:
            current_cost = abs(current_sum)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_coeffs = list(coeffs)
            
        T = L / 10.0
        start_time = time.time()
        
        # Local variables for speed
        w = weights
        c = coeffs
        s = current_sum
        cost = current_cost
        L_val = L
        
        while time.time() - start_time < time_per_run:
            i = random.randint(0, 29)
            old_val = c[i]
            r = random.random()
            if old_val == -1:
                new_val = 0 if r < 0.5 else 1
            elif old_val == 0:
                new_val = -1 if r < 0.5 else 1
            else:
                new_val = -1 if r < 0.5 else 0
                
            delta_sum = (new_val - old_val) * w[i]
            new_sum = s + delta_sum
            
            if new_sum == 0:
                new_cost = L_val
            else:
                new_cost = abs(new_sum)
                
            delta_cost = new_cost - cost
            
            if delta_cost <= 0:
                # Accept
                c[i] = new_val
                s = new_sum
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_coeffs = list(c)
            else:
                if T > 1e-6:
                    # Only compute exp if there's a reasonable chance
                    if delta_cost < T * 50:
                        prob = math.exp(-delta_cost / T)
                        if random.random() < prob:
                            c[i] = new_val
                            s = new_sum
                            cost = new_cost
                            if cost < best_cost:
                                best_cost = cost
                                best_coeffs = list(c)
            
            T *= 0.999
            if T < 1e-6:
                T = 1e-6

    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
