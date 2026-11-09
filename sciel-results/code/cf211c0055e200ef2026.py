import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin vs 90s kill
    
    N = 30
    
    # Precompute LCM and scaled values
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, N + 1))
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = L // i
    
    def compute_sum(c):
        s = 0
        for i in range(N):
            if c[i] != 0:
                s += c[i] * A[i + 1]
        return s
    
    def abs_val(s):
        return abs(s)
    
    best = None
    best_abs = None
    
    # Multiple random restarts with simulated annealing
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 1.0:
            break
        
        # Initialize with random coefficients
        c = [random.choice([-1, 0, 1]) for _ in range(N)]
        # Ensure at least one nonzero
        if all(x == 0 for x in c):
            c[0] = 1
        
        cur = compute_sum(c)
        if cur == 0:
            c[0] = 1
            cur = compute_sum(c)
        
        if best is None or abs_val(cur) < best_abs:
            best_abs = abs_val(cur)
            best = c[:]
        
        # Simulated annealing
        T = 1000.0
        T_min = 1.0
        cooling = 0.95
        steps = 0
        
        while time.time() < deadline - 0.5 and T > T_min:
            steps += 1
            # Pick a random position to change
            j = random.randint(0, N - 1)
            old = c[j]
            
            # Try all possible new values
            candidates = [v for v in (-1, 0, 1) if v != old]
            new = random.choice(candidates)
            
            delta = (new - old) * A[j + 1]
            new_sum = cur + delta
            
            if new_sum == 0:
                # Reject zero sums
                continue
            
            # Accept if better or with probability exp(-delta_abs / T)
            if abs_val(new_sum) < abs_val(cur):
                c[j] = new
                cur = new_sum
                if best is None or abs_val(cur) < best_abs:
                    best_abs = abs_val(cur)
                    best = c[:]
            else:
                # Metropolis criterion
                import math
                prob = math.exp(-abs_val(new_sum) / T)
                if random.random() < prob:
                    c[j] = new
                    cur = new_sum
            
            T *= cooling
        
        # Local search refinement after annealing
        improved = True
        while improved and time.time() < deadline - 0.3:
            improved = False
            for j in range(N):
                old = c[j]
                for nv in (-1, 0, 1):
                    if nv == old:
                        continue
                    ns = cur + (nv - old) * A[j + 1]
                    if ns == 0:
                        continue
                    if abs_val(ns) < abs_val(cur):
                        c[j] = nv
                        cur = ns
                        improved = True
                        if best is None or abs_val(cur) < best_abs:
                            best_abs = abs_val(cur)
                            best = c[:]
    
    # Final local search on best solution
    if best is not None:
        cur = compute_sum(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for j in range(N):
                old = best[j]
                for nv in (-1, 0, 1):
                    if nv == old:
                        continue
                    ns = cur + (nv - old) * A[j + 1]
                    if ns == 0:
                        continue
                    if abs_val(ns) < abs_val(cur):
                        best[j] = nv
                        cur = ns
                        improved = True
    
    return best
