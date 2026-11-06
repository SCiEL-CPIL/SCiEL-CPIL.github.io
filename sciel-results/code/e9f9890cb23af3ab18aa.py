import time
import random
import math

def construct():
    # Calculate LCM of 1..30
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    w = [L // i for i in range(1, 31)]
    
    best_abs = float('inf')
    best_coeffs = None
    
    start_time = time.time()
    deadline = 50.0 # 50 seconds
    
    # Pre-generate random choices? No, just use random.choice
    
    while time.time() - start_time < deadline:
        # Initialize
        c = [random.choice([-1, 0, 1]) for _ in range(30)]
        s = sum(c[i] * w[i] for i in range(30))
        
        if s == 0:
            # Perturb
            c[0] = 1
            s += w[0]
            
        if abs(s) < best_abs:
            best_abs = abs(s)
            best_coeffs = c.copy()
            
        current_s = s
        current_c = c
        temp = 1.0
        
        # Local search
        # Number of steps depends on time left, but fixed is okay if loop is fast
        for step in range(5000):
            if time.time() - start_time > deadline:
                break
                
            k = random.randint(0, 29)
            old_val = current_c[k]
            # Choose a new value different from old
            options = [-1, 0, 1]
            options.remove(old_val)
            new_val = random.choice(options)
            
            delta = (new_val - old_val) * w[k]
            new_s = current_s + delta
            
            if new_s == 0:
                continue
                
            abs_new = abs(new_s)
            abs_cur = abs(current_s)
            
            accept = False
            if abs_new < abs_cur:
                accept = True
                if abs_new < best_abs:
                    best_abs = abs_new
                    best_coeffs = current_c.copy()
            else:
                # Simulated annealing
                # delta_E = abs_new - abs_cur > 0
                # p = exp(-delta_E / temp)
                # Since values are integers and can be large, scaling temp is tricky.
                # Let's use a simpler heuristic: accept worse moves with small probability.
                # Or just greedy with random restarts.
                # Given the ruggedness, pure greedy might get stuck.
                # Let's try a small probability of accepting worse moves.
                if random.random() < 0.01:
                    accept = True
            
            if accept:
                current_s = new_s
                current_c[k] = new_val
                
    return best_coeffs
