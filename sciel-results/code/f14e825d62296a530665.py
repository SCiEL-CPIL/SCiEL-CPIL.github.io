"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    candidates = list(range(20, 201))
    
    def is_compatible(m_new, r_new, existing):
        for (m_old, r_old) in existing:
            g = gcd(m_new, m_old)
            if r_new % g == r_old % g:
                return False
        return True
    
    def compute_score(classes):
        return sum(1.0 / m for m, r in classes)
    
    best_set = []
    best_score = 0.0
    
    # Try multiple random orderings
    for trial in range(300):
        if time.time() > deadline - 5:
            break
        
        order = candidates[:]
        # Sort by 1/m descending with small random perturbation
        order.sort(key=lambda m: -1.0/m + random.random()*0.05)
        
        current_set = []
        current_score = 0.0
        
        for m in order:
            if time.time() > deadline - 2:
                break
            # Try residues in a deterministic order that favors small residues
            for r in range(m):
                if is_compatible(m, r, current_set):
                    current_set.append((m, r))
                    current_score += 1.0/m
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
    
    if not best_set:
        best_set = [(20*j, j-1) for j in range(1, 11)]
    
    return [[m, r] for m, r in best_set]

# EVOLVE-BLOCK-END
