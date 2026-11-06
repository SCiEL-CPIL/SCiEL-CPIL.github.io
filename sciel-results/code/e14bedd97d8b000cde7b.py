import math
import time

def construct():
    deadline = time.time() + 110
    
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        # Find a valid residue r for this m
        for r in range(m):
            valid = True
            for (m2, r2) in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    valid = False
                    break
            if valid:
                selected.append((m, r))
                break
    
    return [[m, r] for m, r in selected]
