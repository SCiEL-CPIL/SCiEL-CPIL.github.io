"""Optimized odd covering system search using precomputed indices and greedy maximum coverage."""
import numpy as np

BASE = 51975

def construct():
    base = BASE
    
    # 1. Get divisors > 1
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    
    # 2. Precompute candidates and their indices
    # Candidate: (m, r, indices)
    candidates = []
    for m in divisors:
        for r in range(m):
            # Precompute indices for this residue class
            # Since m divides base, the length is exactly base // m
            indices = np.arange(r, base, m)
            candidates.append((m, r, indices))
            
    # 3. Sort candidates by max potential coverage (base // m) descending
    # This allows us to prune candidates in the greedy loop
    # Smaller m -> larger base // m
    candidates.sort(key=lambda x: -base // x[0])
    
    # 4. Greedy selection
    remaining = np.ones(base, dtype=bool)
    used_m = set()
    classes = []
    
    # We can use the fact that candidates are sorted by max_gain
    # max_gain for candidate i is base // m_i
    # We can precompute max_gains for faster pruning check
    max_gains = [base // c[0] for c in candidates]
    
    for _ in range(len(divisors)):
        best_gain = 0
        best_idx = -1
        
        for i, (m, r, indices) in enumerate(candidates):
            if m in used_m:
                continue
                
            # Pruning: if the maximum possible gain for this candidate
            # is less than or equal to the best gain found so far,
            # we can skip it. Since the list is sorted by max_gain descending,
            # all subsequent candidates will have <= max_gain.
            if max_gains[i] <= best_gain:
                break
                
            # Calculate actual gain
            gain = remaining[indices].sum()
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_gain == 0 or best_idx == -1:
            break
        
        m, r, indices = candidates[best_idx]
        used_m.add(m)
        classes.append([m, r])
        
        # Update remaining
        remaining[indices] = False
        
    return classes
