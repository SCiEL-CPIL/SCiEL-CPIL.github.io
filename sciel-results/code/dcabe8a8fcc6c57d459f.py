# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Precompute all APs as (start, step, length) or as index lists
_N = 256

def _precompute_aps():
    """Return list of (positions_array) for all APs with length >= 2."""
    aps = []
    for d in range(1, _N):
        for r in range(d):
            pos = np.arange(r, _N, d)
            if len(pos) < 2:
                continue
            # All subarrays of pos
            L = len(pos)
            for i in range(L):
                for j in range(i+1, L+1):
                    aps.append(pos[i:j])
    return aps

# This would be too many (~180k). Let's not precompute all.
# Instead, compute max on the fly per evaluation using the prefix sum method.

def _max_abs_ap_sum(signs):
    n = len(signs)
    max_val = 0
    for d in range(1, n):
        for r in range(d):
            pos = np.arange(r, n, d)
            L = len(pos)
            if L < 2:
                continue
            vals = signs[pos]
            prefix = np.empty(L+1, dtype=np.int32)
            prefix[0] = 0
            np.cumsum(vals, out=prefix[1:])
            # Max subarray sum and min subarray sum
            # Use Kadane-like with running min/max
            running_min = np.minimum.accumulate(prefix)
            running_max = np.maximum.accumulate(prefix)
            max_pos = np.max(prefix - running_min)
            max_neg = np.max(running_max - prefix)
            cur = max(max_pos, max_neg)
            if cur > max_val:
                max_val = cur
    return int(max_val)

def construct():
    deadline = time.time() + 85.0
    n = _N
    rng = random.Random(12345)
    
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int32)
    best_signs = signs.copy()
    best_score = _max_abs_ap_sum(signs)
    
    temp = 5.0
    cool_rate = 0.9995
    
    while time.time() < deadline:
        idx = rng.randrange(n)
        signs[idx] = -signs[idx]
        new_score = _max_abs_ap_sum(signs)
        
        if new_score <= best_score:
            best_signs = signs.copy()
            best_score = new_score
            temp = max(0.5, temp * cool_rate)
        else:
            if rng.random() < np.exp(-(new_score - best_score) / temp):
                temp = max(0.5, temp * cool_rate)
            else:
                signs[idx] = -signs[idx]
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
