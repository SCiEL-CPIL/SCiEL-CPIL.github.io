import time
import random
import numpy as np

def _max_abs_ap_sum(s, target=100):
    n = len(s)
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            seq = s[r::d]
            L = len(seq)
            if L < 2:
                continue
            if L <= max_disc:
                continue
            ps = np.cumsum(seq)
            min_ps = np.min(ps)
            max_ps = np.max(ps)
            min_val = 0 if min_ps > 0 else min_ps
            max_val = 0 if max_ps < 0 else max_ps
            val = max_val - min_val
            if val > max_disc:
                max_disc = val
                if max_disc >= target:
                    return max_disc
    return max_disc

def construct():
    n = 256
    deadline = time.time() + 55.0
    rng = random.Random(171)
    
    best_s = None
    best_val = float('inf')
    
    while time.time() < deadline:
        s = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int32)
        val = _max_abs_ap_sum(s)
        
        if val < best_val:
            best_val = val
            best_s = s.copy()
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip_idx = -1
            best_flip_val = val
            
            for i in range(n):
                old_v = s[i]
                s[i] = -old_v
                
                # Use current val as target for pruning
                new_val = _max_abs_ap_sum(s, target=val)
                
                s[i] = old_v
                
                if new_val < best_flip_val:
                    best_flip_val = new_val
                    best_flip_idx = i
            
            if best_flip_idx != -1:
                s[best_flip_idx] = -s[best_flip_idx]
                val = best_flip_val
                improved = True
                
                if val < best_val:
                    best_val = val
                    best_s = s.copy()
    
    return best_s.tolist()
