import time
import random

def construct() -> dict:
    # We return the Paley tournament on 67 vertices, which is the smallest known
    # valid tournament for the 4-domination property (Schütte's problem k=4).
    # The Paley tournament on a prime p ≡ 3 (mod 4) has the property that
    # every set of 4 vertices has a common dominator when p ≥ 67.
    
    deadline = time.time() + 115  # safety margin before 120s limit
    
    p = 67
    
    # Precompute quadratic residues modulo p
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
        
    # Construct the orientation bits
    # Standard lexicographic order:
    # (0,1), (0,2), ..., (0,p-1), (1,2), (1,3), ..., (1,p-1), ...
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in residues:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': p, 'orientation': orientation}
