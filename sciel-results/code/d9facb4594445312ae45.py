import numpy as np
from bisect import bisect_left
from collections import defaultdict
import time

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # 1. Compute sigma
    sigma = np.ones(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        # Vectorized addition
        sigma[d::d] += d
        
    # 2. Build mapping
    sigma_map = defaultdict(list)
    # We only need to store numbers up to N
    for n in range(1, N + 1):
        s = int(sigma[n])
        sigma_map[s].append(n)
        
    # 3. Search
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5) # Initial seed from prompt
    
    # Iterate b from N down to 1
    # We can skip b if 1/(2*b) >= best_err, because even the best possible match for this b or smaller b cannot beat best_err
    # Actually, for b, min err is ~ 1/(2b).
    # If 1/(2*b) >= best_err, then for this b and all smaller b, we can't improve.
    # So we can break.
    
    for b in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        # Check early stop
        # Min possible error for this b is approx 1/(2*b)
        # If this is larger than our current best, we can stop completely
        if 1.0 / (2.0 * b) >= best_err:
            break
            
        s = int(sigma[b])
        if s not in sigma_map:
            continue
            
        L = sigma_map[s]
        if len(L) < 2:
            # Only one number with this sigma, which is b itself (since b is in L)
            # Or maybe b is not in L? No, b was added.
            # So L = [b]. Error is 0.5.
            # If best_err < 0.5, we can skip.
            if best_err < 0.5:
                continue
            else:
                # Just in case, though 0.5 is unlikely to be best
                pass
        
        target = 1.5 * b
        # Find closest in L
        idx = bisect_left(L, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len(L):
            candidates.append(L[idx])
        if idx > 0:
            candidates.append(L[idx-1])
            
        for a in candidates:
            if a == b:
                continue
            if a > N or a < 1:
                continue
            err = abs(a / b - 1.5)
            if err < best_err:
                best_err = err
                best_a = a
                best_b = b
                
    return [int(best_a), int(best_b)]
