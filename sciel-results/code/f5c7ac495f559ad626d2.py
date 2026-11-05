import time
from fractions import Fraction
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Compute LCM(2, 3, ..., 48)
    L = 1
    for d in denominators:
        L = L // gcd(L, d) * d

    # We want to find a subset S such that sum_{i in S} 1/d_i = 1/1806
    # Then the answer is all indices NOT in S
    # In integer terms: sum_{i in S} (L / d_i) = L / 1806
    
    target = L // 1806  # This should be an integer since 1806 | L
    
    # Verify
    assert L % 1806 == 0, f"L={L} not divisible by 1806"
    
    # We need to find subset of {L/2, L/3, ..., L/48} that sums to target
    # These are very large numbers, but target is much smaller
    # L/1806 is relatively small compared to the individual terms
    
    # Sort the weights
    weights = [L // d for d in denominators]
    
    # Only consider weights that are <= target (since we can't use larger ones)
    # Actually, we need exact sum, so we can use any combination
    # But weights larger than target can't be part of the solution (unless we don't use them)
    
    # Filter: only weights <= target can be in the subset
    # Actually no - we need sum to be exactly target, so any weight > target can't be included
    valid_indices = [i for i in range(n) if weights[i] <= target]
    valid_weights = [weights[i] for i in valid_indices]
    valid_orig = valid_indices
    
    # Sort by weight
    order = sorted(range(len(valid_weights)), key=lambda i: -valid_weights[i])
    sorted_w = [valid_weights[i] for i in order]
    sorted_orig = [valid_orig[i] for i in order]
    
    m = len(sorted_w)
    
    # Suffix sums
    suffix = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sorted_w[i]
    
    # B&B to find subset summing to exactly target
    found = [False]
    found_sel = [[]]
    
    def bb():
        nonlocal found, found_sel
        stack = [(0, 0, [])]
        
        while stack:
            idx, curr, sel = stack.pop()
            
            if time.time() > deadline - 3:
                break
            
            if found[0]:
                break
            
            # Prune: if curr > target, skip
            if curr > target:
                continue
            
            # Prune: if curr + suffix[idx] < target, can't reach target
            if curr + suffix[idx] < target:
                continue
            
            if idx == m:
                if curr == target:
                    found[0] = True
                    found_sel[0] = list(sel)
                    break
                continue
            
            # Skip
            stack.append((idx + 1, curr, sel))
            
            # Take
            new_curr = curr + sorted_w[idx]
            if new_curr <= target:
                sel.append(idx)
                stack.append((idx + 1, new_curr, sel))
                sel.pop()
    
    bb()
    
    if found[0]:
        # The subset to REMOVE is found_sel[0]
        remove_set = set()
        for i in found_sel[0]:
            remove_set.add(sorted_orig[i])
        
        # The answer is all indices NOT in remove_set
        selected = [i for i in range(n) if i not in remove_set]
    else:
        # Fallback: use greedy
        selected = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                selected.append(k)
                s += Fraction(1, denominators[k])
    
    # Verify
    total = Fraction(0)
    for idx in selected:
        total += Fraction(1, denominators[idx])
    
    if total >= 1 or total == 0:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel
    
    return selected
