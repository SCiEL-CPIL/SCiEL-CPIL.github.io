"""Optimized odd covering system search using bitset greedy maximum coverage."""
import numpy as np

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

def construct():
    base = BASE
    # Precompute all divisors > 1 of 51975
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Precompute masks for all possible classes (m, r)
    # mask is an integer where bit i is 1 if i % m == r
    # We store them in a list of tuples (m, r, mask)
    all_classes = []
    for m in divisors:
        for r in range(m):
            # Create mask for residue r modulo m
            # We can do this by iterating or using a trick
            # Since base is small (51975), we can just build it
            mask = 0
            # Optimization: build mask using bit shifts or simple loop
            # Given base ~ 50k, a simple loop is fine for precomputation
            for i in range(r, base, m):
                mask |= (1 << i)
            all_classes.append((m, r, mask))
            
    # Sort classes by potential coverage (number of 1s in mask) descending
    # This helps greedy find good candidates earlier, though it checks all anyway
    # But sorting by size might help if we break early or just for deterministic order
    # Actually, greedy must check all unused to find max gain.
    # We can sort by m ascending (larger coverage) to potentially find high gain faster in some heuristics,
    # but strictly we must compare all.
    
    # Initial uncovered set: all bits 0 to base-1 are 1
    # mask_all = (1 << base) - 1
    uncovered = (1 << base) - 1
    used_moduli = set()
    classes = []
    
    # Greedy loop
    # We have at most len(divisors) classes to pick
    while True:
        best_gain = 0
        best_class = None
        
        for m, r, mask in all_classes:
            if m in used_moduli:
                continue
            # Calculate gain: number of bits set in (mask & uncovered)
            # Python int.bit_count() is available in 3.10+
            # Assuming Python 3.10+ as per standard modern environments
            gain = (mask & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_class = (m, r, mask)
                
        if best_class is None or best_gain == 0:
            break
            
        m, r, mask = best_class
        classes.append([m, r])
        used_moduli.add(m)
        # Update uncovered: clear bits that are now covered
        uncovered &= ~mask
        
        # Optional: Early termination if uncovered is 0
        if uncovered == 0:
            break
            
    return classes
