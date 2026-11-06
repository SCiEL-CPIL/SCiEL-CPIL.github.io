import random
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (limit 90s)
    
    N = 30
    # Calculate LCM of 1..30
    L = reduce(lambda a, b: a * b // gcd(a, b), range(1, N + 1))
    # a_i = L / i
    a = [L // i for i in range(1, N + 1)]
    
    best_abs_K = float('inf')
    best_coeff = None
    
    def get_K(c):
        s = 0
        for i in range(N):
            if c[i] == 1:
                s += a[i]
            elif c[i] == -1:
                s -= a[i]
        return s
    
    def local_search(c_init):
        nonlocal best_abs_K, best_coeff
        
        c = c_init[:]
        K = get_K(c)
        
        # Ensure nonzero
        if K == 0:
            for i in range(N):
                old = c[i]
                if old == 0:
                    c[i] = 1
                else:
                    c[i] = -old
                K = get_K(c)
                if K != 0:
                    break
            if K == 0:
                return
        
        if abs(K) < best_abs_K:
            best_abs_K = abs(K)
            best_coeff = c[:]
        
        # Hill climbing with random kicks
        steps = 0
        while time.time() < deadline:
            steps += 1
            if steps > 50000: # Hard cap per restart to allow more restarts
                break
                
            # Try a random move
            i = random.randint(0, N - 1)
            old_val = c[i]
            
            # Determine possible new values
            if old_val == 0:
                candidates = [-1, 1]
            elif old_val == 1:
                candidates = [-1, 0]
            else: # -1
                candidates = [0, 1]
                
            random.shuffle(candidates)
            
            improved = False
            for new_val in candidates:
                # Calculate delta
                if new_val == 1:
                    delta = a[i] - (old_val * a[i])
                elif new_val == -1:
                    delta = -a[i] - (old_val * a[i])
                else:
                    delta = - (old_val * a[i])
                    
                K_new = K + delta
                
                if K_new == 0:
                    continue # Invalid
                
                if abs(K_new) < abs(K):
                    c[i] = new_val
                    K = K_new
                    improved = True
                    
                    if abs(K) < best_abs_K:
                        best_abs_K = abs(K)
                        best_coeff = c[:]
                    break
            
            if not improved:
                # Kick: flip 2 random bits
                if random.random() < 0.2:
                    i1 = random.randint(0, N - 1)
                    i2 = random.randint(0, N - 1)
                    if i1 == i2:
                        i2 = (i2 + 1) % N
                    
                    v1 = c[i1]
                    v2 = c[i2]
                    
                    # Try flipping to random valid
                    new_v1 = random.choice([-1, 0, 1])
                    new_v2 = random.choice([-1, 0, 1])
                    
                    if new_v1 == v1 and new_v2 == v2:
                        continue
                         
                    c[i1] = new_v1
                    c[i2] = new_v2
                    K = get_K(c)
                    if K == 0:
                        # Revert
                        c[i1] = v1
                        c[i2] = v2
                        K = get_K(c)
                        continue
                    if abs(K) < best_abs_K:
                        best_abs_K = abs(K)
                        best_coeff = c[:]

    # Run multiple restarts
    while time.time() < deadline:
        c_init = [random.choice([-1, 0, 1]) for _ in range(N)]
        local_search(c_init)
        
    if best_coeff is None:
        # Fallback
        best_coeff = [0] * N
        best_coeff[0] = 1
        best_coeff[1] = -1
        
    return best_coeff
