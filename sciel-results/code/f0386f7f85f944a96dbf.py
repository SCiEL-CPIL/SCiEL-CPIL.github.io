import numpy as np
import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute 4-APs
    # aps_containing[i] = list of (j, k, l)
    aps_containing = [[] for _ in range(N)]
    
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                i0 = a - 1
                i1 = b - 1
                i2 = c - 1
                i3 = e - 1
                
                aps_containing[i0].append((i1, i2, i3))
                aps_containing[i1].append((i0, i2, i3))
                aps_containing[i2].append((i0, i1, i3))
                aps_containing[i3].append((i0, i1, i2))

    def count_mono(x):
        count = 0
        # Vectorized-ish check? 
        # Just loop over APs.
        # We can reuse the generation logic or precompute the list of APs as tuples of indices.
        # Let's precompute a flat list of APs for counting.
        pass # We'll use the aps_containing for delta, but need full list for count?
        # Actually, count_mono is only called at init and maybe end. 
        # We can compute it by summing 1 if all 4 are same.
        # To be safe and fast, let's precompute the list of all APs as (i0, i1, i2, i3).
        
    # Precompute all APs list
    all_aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                all_aps.append((a-1, b-1, c-1, e-1))
                
    def count_mono(x):
        count = 0
        for ap in all_aps:
            if x[ap[0]] == x[ap[1]] == x[ap[2]] == x[ap[3]]:
                count += 1
        return count

    def get_delta(x, i):
        old_val = x[i]
        new_val = 1 - old_val
        delta = 0
        for j, k, l in aps_containing[i]:
            if x[j] == new_val and x[k] == new_val and x[l] == new_val:
                delta += 1
            if x[j] == old_val and x[k] == old_val and x[l] == old_val:
                delta -= 1
        return delta

    best_score = float('inf')
    best_x = None
    
    # Parameters for SA
    T0 = 2.0
    cooling_rate = 0.99
    sa_steps = 5000
    
    for restart in range(30):
        if time.time() > deadline - 10.0:
            break
            
        # Random initialization
        x = np.random.randint(0, 2, N)
        current_score = count_mono(x)
        
        # Phase 1: Greedy Local Search
        improved = True
        while improved:
            if time.time() > deadline - 10.0:
                break
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline - 10.0:
                    break
                delta = get_delta(x, i)
                if delta < 0:
                    x[i] = 1 - x[i]
                    current_score += delta
                    improved = True
        
        # Phase 2: Simulated Annealing
        T = T0
        for _ in range(sa_steps):
            if time.time() > deadline - 10.0:
                break
            i = random.randint(0, N - 1)
            delta = get_delta(x, i)
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta < 0:
                x[i] = 1 - x[i]
                current_score += delta
            else:
                if delta == 0:
                    # Accept neutral moves to explore
                    if random.random() < 0.5:
                        x[i] = 1 - x[i]
                        # current_score unchanged
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        x[i] = 1 - x[i]
                        current_score += delta
            
            T *= cooling_rate
            
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
    if best_x is None:
        return [(i // 2) % 2 for i in range(N)]
        
    return best_x.tolist()
