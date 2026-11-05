import math
import time

def construct():
    # We want to maximize sum(1/m) for distinct m in [20, 200]
    # with pairwise disjoint residues.
    # Greedy approach: iterate m from 20 to 200.
    # For each m, check if there is a valid residue r.
    # If yes, add (m, r) to the solution.
    
    selected = []
    
    # Precompute gcds? Not necessary, math.gcd is fast.
    
    for m in range(20, 201):
        # Check if there is a valid residue for m
        # valid_r[r] is True if r is a candidate
        valid_r = [True] * m
        
        # Check against all previously selected (m_j, r_j)
        for m_j, r_j in selected:
            g = math.gcd(m, m_j)
            target = r_j % g
            # Forbid r such that r % g == target
            # r = target, target + g, target + 2g, ... < m
            for r in range(target, m, g):
                valid_r[r] = False
        
        # Find the first valid residue
        found = False
        for r in range(m):
            if valid_r[r]:
                selected.append([m, r])
                found = True
                break
        
        if not found:
            # No valid residue for this m, skip it
            pass
            
    return selected
