"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _find_max_classes(max_mod=128, deadline=None):
    """
    Multi-start randomized greedy search for maximum pairwise-disjoint residue classes.
    
    Strategy:
    1. Generate candidate moduli (2..max_mod).
    2. Try many random orderings of these moduli.
    3. For each ordering, greedily add a modulus if there exists a residue that is 
       compatible with all currently selected classes.
    4. Keep the best result found within the time limit.
    """
    if deadline is None:
        deadline = time.time() + 80  # safety margin
    
    best_set = []
    best_size = 0
    
    # Pre-compute gcds for all pairs of moduli up to max_mod
    gcd_table = [[0]*(max_mod+1) for _ in range(max_mod+1)]
    for m1 in range(2, max_mod+1):
        for m2 in range(2, max_mod+1):
            gcd_table[m1][m2] = gcd(m1, m2)
            
    # Candidate moduli
    all_moduli = list(range(2, max_mod+1))
    
    start_time = time.time()
    max_iters = 10000 # Cap iterations to prevent infinite loop if time check is slow
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        # Shuffle moduli
        moduli = all_moduli[:]
        random.shuffle(moduli)
        
        current_set = []
        
        for m in moduli:
            # Check coprime constraint first
            compatible_moduli = True
            for r_old, m_old in current_set:
                if gcd_table[m][m_old] == 1:
                    compatible_moduli = False
                    break
            
            if not compatible_moduli:
                continue
            
            # Now find a residue r for m that works with all current_set
            found_r = -1
            for r in range(m):
                ok = True
                for r_old, m_old in current_set:
                    g = gcd_table[m][m_old]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            
            if found_r != -1:
                current_set.append([found_r, m])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = [list(c) for c in current_set]
    
    return best_set

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
