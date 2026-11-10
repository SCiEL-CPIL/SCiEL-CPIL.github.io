import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    # We want to maximize sum(1/m). Smaller m is better.
    # We need pairwise disjoint residue classes.
    # Strategy: Greedy addition of smallest moduli, finding a valid residue for each.
    
    selected = []  # List of (m, r)
    
    # Candidate moduli from 20 to 200
    candidates = list(range(20, 201))
    
    for m in candidates:
        if time.time() > deadline:
            break
            
        found_residue = None
        # Try all possible residues for this modulus
        # Optimization: We can break early if we find one
        for r in range(m):
            valid = True
            for (m2, r2) in selected:
                g = gcd(m, m2)
                if r % g == r2 % g:
                    valid = False
                    break
            if valid:
                found_residue = r
                break
                
        if found_residue is not None:
            selected.append((m, found_residue))
            
    # If we found nothing, return a single valid class (always possible)
    if not selected:
        return [[20, 0]]
        
    return [[m, r] for m, r in selected]
