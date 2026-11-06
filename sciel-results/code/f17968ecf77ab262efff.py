import math
import time
import random
from functools import lru_cache

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = _gcd(m1, m2)
    if g == 1:
        return False  # always intersect
    return (r2 - r1) % g != 0

def _can_add(r_new, m_new, existing, gcd_table):
    """Check if new (r_new, m_new) is disjoint from all existing classes."""
    for (r, m) in existing:
        g = gcd_table.get((m_new, m), 1)
        if g == 1:
            return False
        if (r_new - r) % g == 0:
            return False
    return True

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin for 90s limit
    
    # All moduli from 2 to 128
    all_moduli = list(range(2, 129))
    
    # Precompute gcd table for all pairs
    gcd_table = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = _gcd(m1, m2)
    
    best_result = None
    best_count = 0
    
    # Strategy: Greedy with multiple random orderings
    # Key insight: We need moduli that are NOT coprime (gcd > 1) with each other
    # to have a chance of being disjoint. So we should focus on non-coprime sets.
    
    # First, let's try a pure greedy approach with various orderings
    # We'll try to maximize the count
    
    # Precompute for each modulus, which other moduli it shares a factor with
    # This helps us pick promising orderings
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Create a random ordering
        # Bias toward moduli that share factors with many others
        # For now, just use random shuffles with different seeds
        
        random.seed(attempt)
        perm = all_moduli[:]
        random.shuffle(perm)
        
        selected = []
        
        for m in perm:
            if time.time() > deadline:
                break
            
            # Check if m is coprime to any already selected modulus
            # If yes, we cannot add m (since disjoint requires gcd > 1 and residue mismatch)
            # Actually, if gcd(m, selected_m) == 1, they always intersect, so skip
            skip = False
            for (r, sm) in selected:
                if _gcd(m, sm) == 1:
                    skip = True
                    break
            if skip:
                continue
            
            # Try to find a residue for m that works
            # Use random order for residues to break symmetry
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                if _can_add(r, m, selected, gcd_table):
                    selected.append((r, m))
                    found = True
                    break
            
            # If not found, skip this modulus (greedy)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = selected[:]
        
        # Early exit if we find a very good solution
        # The maximum possible is bounded by the size of the largest non-coprime subset
        # For moduli 2..128, the evens give 63 moduli. But not all can be disjoint.
        # Let's see what we can achieve.
        if best_count >= 50:
            break
    
    # If greedy didn't find anything good, try a more structured approach
    # Focus on powers of 2 or other structured sets
    
    if best_result is None or best_count < 10:
        # Structured fallback: Use moduli that are powers of 2
        # 2, 4, 8, 16, 32, 64, 128 -> 7 classes
        # With careful residue choice, these can be pairwise disjoint
        structured = []
        for k in range(7):
            m = 1 << (k + 1)
            if m <= 128:
                r = (1 << k) - 1  # e.g., 1, 3, 7, 15, 31, 63, 127
                structured.append((r, m))
        
        # Verify
        valid = True
        for i in range(len(structured)):
            for j in range(i + 1, len(structured)):
                if not _disjoint(structured[i][0], structured[i][1], structured[j][0], structured[j][1]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(structured) > best_count:
            best_result = structured
            best_count = len(structured)
    
    # Final fallback
    if not best_result:
        best_result = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    
    return [[r, m] for r, m in best_result]
