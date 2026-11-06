import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // i for i in range(1, 31)]
    
    best_N = None
    best_c = None
    
    def evaluate(c):
        return sum(c[i] * weights[i] for i in range(30))
    
    def local_search(initial_c):
        nonlocal best_N, best_c
        c = initial_c[:]
        current_N = evaluate(c)
        
        if current_N != 0:
            if best_N is None or abs(current_N) < abs(best_N):
                best_N = current_N
                best_c = c[:]
            if abs(current_N) == 1:
                return c
        
        while time.time() < deadline:
            improved = False
            indices = list(range(30))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                old_val = c[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    
                    delta = (new_val - old_val) * weights[i]
                    new_N = current_N + delta
                    
                    if new_N == 0:
                        continue
                    
                    if abs(new_val - old_val) == 1 and abs(new_N) < abs(current_N):
                        # This condition is redundant if we just check abs(new_N) < abs(current_N)
                        pass

                    if abs(new_N) < abs(current_N):
                        c[i] = new_val
                        current_N = new_N
                        improved = True
                        
                        if best_N is None or abs(current_N) < abs(best_N):
                            best_N = current_N
                            best_c = c[:]
                        if abs(current_N) == 1:
                            return c
                        break
            
            if not improved:
                num_changes = random.randint(1, 3)
                for _ in range(num_changes):
                    idx = random.randint(0, 29)
                    c[idx] = random.randint(-1, 1)
                current_N = evaluate(c)
                
                if current_N != 0:
                    if best_N is None or abs(current_N) < abs(best_N):
                        best_N = current_N
                        best_c = c[:]
                    if abs(current_N) == 1:
                        return c
        
        return c

    seed_c = [0] * 28 + [1, -1]
    local_search(seed_c)
    
    while time.time() < deadline:
        c = [random.randint(-1, 1) for _ in range(30)]
        local_search(c)
        
        if best_N is not None and abs(best_N) == 1:
            break

    if best_c is None:
        return seed_c
    
    return best_c
