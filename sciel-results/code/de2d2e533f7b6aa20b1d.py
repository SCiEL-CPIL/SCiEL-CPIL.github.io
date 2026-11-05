import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or numpy array of +1/-1.
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    
    # Iterate over step d
    for d in range(1, n + 1):
        # For each residue class r in 0..d-1
        for r in range(min(d, n)):
            # Extract subsequence: indices r, r+d, r+2d, ...
            sub = signs_arr[r::d]
            m = len(sub)
            if m <= 1:
                if 1 > max_abs:
                    max_abs = 1
                continue
            
            # Calculate max subarray sum and min subarray sum
            cum = np.cumsum(sub)
            cum_ext = np.concatenate(([0], cum))
            
            min_prefix = np.minimum.accumulate(cum_ext)
            max_prefix = np.maximum.accumulate(cum_ext)
            
            # Max subarray sum
            max_sub = np.max(cum_ext[1:] - min_prefix[:-1])
            # Min subarray sum
            min_sub = np.min(cum_ext[1:] - max_prefix[:-1])
            
            current_max = max(abs(max_sub), abs(min_sub))
            if current_max > max_abs:
                max_abs = current_max
                
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 256
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Thue-Morse
    tm = []
    for i in range(n):
        cnt = bin(i).count('1')
        tm.append(1 if cnt % 2 == 0 else -1)
    candidates.append(tm)
    
    # Walsh-Hadamard (Rademacher functions)
    for k in [1, 2, 3, 4, 5, 6, 7]:
        wh = []
        for i in range(n):
            wh.append(1 if (i >> k) & 1 == 0 else -1)
        candidates.append(wh)
        
    # Random
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        rng = random.Random(seed)
        rand = [rng.choice((-1, 1)) for _ in range(n)]
        candidates.append(rand)
        
    # Sine wave
    for freq in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        sine = [1 if math.sin(2 * math.pi * freq * i / n) >= 0 else -1 for i in range(n)]
        candidates.append(sine)

    # Evaluate and pick best
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    for c in candidates:
        if time.time() > deadline - 10:
            break
        disc = _max_ap_discrepancy(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]

    # 2. Local Search (Hill Climbing)
    signs = best_signs[:]
    current_disc = best_disc
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 2:
        improved = False
        
        # Find worst AP
        worst_d = 1
        worst_r = 0
        worst_val = 0
        worst_sign = 1 # 1 for max positive, -1 for min negative
        
        signs_arr = np.asarray(signs, dtype=np.int8)
        
        for d in range(1, n + 1):
            for r in range(min(d, n)):
                sub = signs_arr[r::d]
                m = len(sub)
                if m <= 1:
                    continue
                cum = np.cumsum(sub)
                cum_ext = np.concatenate(([0], cum))
                min_prefix = np.minimum.accumulate(cum_ext)
                max_prefix = np.maximum.accumulate(cum_ext)
                
                max_sub = np.max(cum_ext[1:] - min_prefix[:-1])
                min_sub = np.min(cum_ext[1:] - max_prefix[:-1])
                
                if abs(max_sub) > abs(worst_val):
                    worst_val = max_sub
                    worst_d = d
                    worst_r = r
                    worst_sign = 1
                elif abs(min_sub) > abs(worst_val):
                    worst_val = min_sub
                    worst_d = d
                    worst_r = r
                    worst_sign = -1
        
        # Try flipping signs in the worst AP
        indices = list(range(worst_r, n, worst_d))
        
        for idx in indices:
            if time.time() > deadline - 2:
                break
            
            old_val = signs[idx]
            
            # Heuristic: Only try flipping if the sign matches the "bad" direction
            if worst_sign == 1 and old_val == -1:
                continue 
            if worst_sign == -1 and old_val == 1:
                continue 

            signs[idx] = -old_val
            new_disc = _max_ap_discrepancy(signs)
            
            if new_disc < current_disc:
                current_disc = new_disc
                improved = True
                no_improve_count = 0
            else:
                signs[idx] = old_val
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Kick: random flip
                idx = random.randint(0, n - 1)
                signs[idx] = -signs[idx]
                new_disc = _max_ap_discrepancy(signs)
                if new_disc < current_disc:
                    current_disc = new_disc
                    no_improve_count = 0
                else:
                    # Revert if it's worse than the best we've ever seen
                    if new_disc > best_disc:
                        signs[idx] = -signs[idx]
                    else:
                        current_disc = new_disc
                        no_improve_count = 0

    return signs
