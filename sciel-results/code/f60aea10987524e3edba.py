import time
import random
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Precompute LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, divisors)
    
    def random_multiple(d, low, high):
        if low > high:
            return None
        k_min = (low + d - 1) // d
        k_max = high // d
        if k_min > k_max:
            return None
        k = random.randint(k_min, k_max)
        return k * d

    best_solution = None
    best_range = float('inf')
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize state
        center = random.randint(100, 10000)
        x = [0] * 10
        used = set()
        valid_init = True
        for i in range(10):
            d = divisors[i]
            found = False
            for offset in range(0, 1000, d):
                for val in [center - offset, center + offset]:
                    if val < 1: continue
                    if val % d == 0 and val not in used:
                        x[i] = val
                        used.add(val)
                        found = True
                        break
                if found:
                    break
            if not found:
                x[i] = random_multiple(d, 1, max_val)
                if x[i] in used:
                    valid_init = False
                    break
                used.add(x[i])
        
        if not valid_init:
            continue
            
        current_range = max(x) - min(x)
        if current_range < best_range:
            best_range = current_range
            best_solution = x[:]
            
        if best_range <= 10:
            break

        # Simulated Annealing
        T = 200.0
        cooling = 0.995
        steps = 20000
        for step in range(steps):
            if time.time() > deadline:
                break
                
            i = random.randint(0, 9)
            old_val = x[i]
            d = divisors[i]
            
            median_val = sorted(x)[5]
            low = max(1, median_val - 300)
            high = min(max_val, median_val + 300)
            
            new_val = random_multiple(d, low, high)
            if new_val is None or new_val == old_val:
                continue
            if new_val in used and new_val != old_val:
                continue
            
            x[i] = new_val
            if len(set(x)) < 10:
                x[i] = old_val
                continue
                
            new_range = max(x) - min(x)
            delta = new_range - current_range
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_range = new_range
                used.remove(old_val)
                used.add(new_val)
                if current_range < best_range:
                    best_range = current_range
                    best_solution = x[:]
            else:
                x[i] = old_val
                
            T *= cooling
            if T < 0.5:
                break
        
    if best_solution:
        if len(set(best_solution)) == 10 and all(best_solution[i] % divisors[i] == 0 for i in range(10)):
            return best_solution
            
    fallback = [ (i+1) * L for i in range(10) ]
    return fallback
