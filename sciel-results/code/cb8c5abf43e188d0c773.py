import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 30
    
    def compute_sum_int(coeff, L):
        s = 0
        for i, c in enumerate(coeff, 1):
            if c != 0:
                s += c * (L // i)
        return s
    
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    best_coeff = None
    best_abs = None
    
    random.seed(42)
    
    num_initial = 50000
    for _ in range(num_initial):
        if time.time() > deadline:
            break
        coeff = [random.choice([-1, 0, 1]) for _ in range(n)]
        s = compute_sum_int(coeff, L)
        if s != 0:
            abs_s = abs(s)
            if best_abs is None or abs_s < best_abs:
                best_abs = abs_s
                best_coeff = coeff[:]
    
    if best_coeff is not None:
        current = best_coeff[:]
        current_abs = best_abs
        no_improve = 0
        max_no_improve = 5000
        
        while no_improve < max_no_improve and time.time() < deadline:
            no_improve += 1
            i = random.randint(0, n - 1)
            old_val = current[i]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            current[i] = new_val
            s = compute_sum_int(current, L)
            if s != 0:
                abs_s = abs(s)
                if abs_s < best_abs:
                    best_abs = abs_s
                    best_coeff = current[:]
                    no_improve = 0
                else:
                    if random.random() < 0.1:
                        pass
                    else:
                        current[i] = old_val
            else:
                current[i] = old_val
    
    if best_coeff is not None:
        return best_coeff
    
    coeff = [0] * n
    coeff[0] = 1
    return coeff
