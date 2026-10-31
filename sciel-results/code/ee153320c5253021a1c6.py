import numpy as np
import random

N = 50

def is_sidon(S):
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def greedy_sidon(order):
    S = []
    sums = set()
    for idx in order:
        # Check if adding idx violates Sidon property
        valid = True
        for existing in S:
            s = idx + existing
            if s in sums:
                valid = False
                break
            # Also check self-sum if idx not in S yet? 
            # The check `s in sums` covers sums of existing pairs.
            # We need to check if `idx+idx` is in sums? 
            # Yes, sums contains sums of pairs from S.
            # If we add idx, new sums are idx+idx and idx+existing.
            # We must ensure none of these are in `sums` (existing sums) 
            # AND none of the new sums collide with each other.
            # Since we check `idx+existing` against `sums`, and 
            # `idx+idx` against `sums`.
            # Do new sums collide with each other?
            # `idx+idx` vs `idx+existing`: equal only if existing == idx (not in S).
            # `idx+existing1` vs `idx+existing2`: equal only if existing1 == existing2.
            # So we just need to check against existing `sums`.
            
        if not valid:
            continue
            
        # Also check if idx+idx is in sums
        if (idx + idx) in sums:
            continue
            
        # Add idx
        S.append(idx)
        sums.add(idx + idx)
        for existing in S[:-1]: # existing elements before idx
            sums.add(idx + existing)
            
    return S

def find_max_sidon_greedy(N, max_restarts=2000):
    best_set = []
    indices = list(range(N))
    
    for _ in range(max_restarts):
        random.shuffle(indices)
        S = greedy_sidon(indices)
        if len(S) > len(best_set):
            best_set = S
            
    return best_set

def construct():
    # Find largest Sidon set using greedy random restarts
    S = find_max_sidon_greedy(N)
    if not S:
        S = [0]
        
    h = np.zeros(N)
    for i in S:
        h[i] = 1.0 / len(S)
        
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)
