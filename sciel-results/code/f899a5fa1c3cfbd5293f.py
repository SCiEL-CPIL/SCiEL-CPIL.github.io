import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Calculate LCM of 1 to 30
    L = 1
    for i in range(1, 31):
        L = math.lcm(L, i)
        
    # Precompute weights
    W = [0] * 30
    for i in range(30):
        W[i] = L // (i + 1)
        
    best_N = None
    best_c = None
    best_abs_N = float('inf')
    
    def evaluate(c):
        N = 0
        for i in range(30):
            if c[i] != 0:
                N += c[i] * W[i]
        return N
        
    # Initial seed solution
    c_seed = [0] * 28 + [1, -1]
    N_seed = evaluate(c_seed)
    if N_seed != 0:
        best_N = N_seed
        best_c = c_seed[:]
        best_abs_N = abs(N_seed)
    else:
        # Fallback if seed is zero (shouldn't happen)
        best_abs_N = float('inf')

    def local_search(c_init):
        c = c_init[:]
        N = evaluate(c)
        if N == 0:
            # Perturb to get non-zero
            idx = random.randrange(30)
            c[idx] = 1
            N = evaluate(c)
            if N == 0:
                c[idx] = -1
                N = evaluate(c)
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(30))
            random.shuffle(indices)
            for i in indices:
                current_val = c[i]
                # Try changing to other values in {-1, 0, 1}
                for new_val in (-1, 0, 1):
                    if new_val == current_val:
                        continue
                    delta = (new_val - current_val) * W[i]
                    new_N = N + delta
                    if new_N == 0:
                        continue
                    if abs(new_N) < abs(N):
                        c[i] = new_val
                        N = new_N
                        improved = True
                        break
        return c, N

    # Run multiple random restarts
    while time.time() < deadline:
        # Random initialization
        c_rand = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        # Run local search
        c_res, N_res = local_search(c_rand)
        
        if N_res != 0 and abs(N_res) < best_abs_N:
            best_abs_N = abs(N_res)
            best_N = N_res
            best_c = c_res[:]
            
        # If we found the theoretical minimum (1), we can stop
        if best_abs_N == 1:
            break

    return best_c
