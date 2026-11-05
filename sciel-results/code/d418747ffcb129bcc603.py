import time
import math
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute the maximum absolute signed sum on ANY arithmetic progression
    contained in [1, n].
    """
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int32)
    max_abs = 0
    
    # Iterate over all possible steps d
    for d in range(1, n + 1):
        # Iterate over all possible starts r (0-indexed)
        for r in range(min(d, n)):
            # Extract the AP: r, r+d, r+2d, ...
            # Using slicing is efficient in NumPy
            sub = signs_arr[r::d]
            m = len(sub)
            if m <= 1:
                if 1 > max_abs:
                    max_abs = 1
                continue
            
            # Compute prefix sums to find max subarray sum
            # Max subarray sum = max(prefix[i] - min_prefix[j]) for j < i
            cum = np.cumsum(sub)
            cum_ext = np.concatenate(([0], cum))
            
            # Min prefix up to i-1
            min_prefix = np.minimum.accumulate(cum_ext)
            # Max prefix up to i-1
            max_prefix = np.maximum.accumulate(cum_ext)
            
            # Max sum ending at i is cum_ext[i] - min_prefix[i]
            # We want max over all i of (cum_ext[i] - min_prefix[i])
            # Note: min_prefix[i] includes cum_ext[i] itself in the accumulation?
            # np.minimum.accumulate([0, c1, c2...])
            # min_prefix[0] = 0
            # min_prefix[1] = min(0, c1)
            # We need min of prefixes BEFORE i.
            # So we should use min_prefix[:-1] and cum_ext[1:]
            
            # Max positive sum
            max_pos = np.max(cum_ext[1:] - min_prefix[:-1])
            # Max negative sum (min subarray sum)
            min_neg = np.min(cum_ext[1:] - max_prefix[:-1])
            
            current_max = max(abs(max_pos), abs(min_neg))
            if current_max > max_abs:
                max_abs = current_max
                
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 256
    
    def evaluate(signs):
        return _max_ap_discrepancy(signs)
    
    # Initial candidates
    candidates = []
    
    # 1. Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)
    
    # 2. Van der Corput (low discrepancy sequence)
    def van_der_corput(n, base=2):
        vdc = [0] * n
        for i in range(n):
            f = 1.0 / base
            x = 0.0
            j = i
            while j > 0:
                x += f * (j % base)
                j //= base
                f /= base
            vdc[i] = 1 if x < 0.5 else -1
        return vdc
    
    vdc = van_der_corput(n)
    candidates.append(vdc)
    
    # 3. Random seeds
    rng = random.Random(42)
    for _ in range(10):
        candidates.append([rng.choice((-1, 1)) for _ in range(n)])
    
    best_signs = candidates[0][:]
    best_disc = float('inf')
    
    # Evaluate initial candidates
    for c in candidates:
        if time.time() > deadline - 30:
            break
        disc = evaluate(c)
        if disc < best_disc:
            best_disc = disc
            best_signs = c[:]
            
    # Simulated Annealing
    # Parameters
    T_start = 5.0
    T_min = 0.01
    alpha = 0.995  # Cooling rate
    
    # We will run multiple restarts
    # Each restart tries to improve upon the current best
    restarts = 50
    rng_sa = random.Random(123)
    
    for r in range(restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize for this restart
        # Start from best found so far, but perturb slightly to escape local optima
        signs = best_signs[:]
        # Perturb 5% of bits
        for _ in range(n // 20):
            idx = rng_sa.randint(0, n - 1)
            signs[idx] = -signs[idx]
            
        current_disc = evaluate(signs)
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs[:]
            
        T = T_start
        no_improve_steps = 0
        max_no_improve = 2000
        
        while T > T_min and time.time() < deadline - 2 and no_improve_steps < max_no_improve:
            # Pick a random index to flip
            idx = rng_sa.randint(0, n - 1)
            
            # Flip
            signs[idx] = -signs[idx]
            new_disc = evaluate(signs)
            delta = new_disc - current_disc
            
            # Acceptance criterion
            if delta < 0 or rng_sa.random() < math.exp(-delta / T):
                # Accept
                current_disc = new_disc
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Reject, revert
                signs[idx] = -signs[idx]
                no_improve_steps += 1
                
            T *= alpha
            
    # Final Polish: Hill Climbing
    signs = best_signs[:]
    current_disc = evaluate(signs)
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for idx in range(n):
            signs[idx] = -signs[idx]
            new_disc = evaluate(signs)
            if new_disc < current_disc:
                current_disc = new_disc
                best_disc = new_disc
                best_signs = signs[:]
                improved = True
                break
            else:
                signs[idx] = -signs[idx] # Revert
    
    return best_signs
