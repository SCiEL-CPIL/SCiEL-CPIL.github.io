import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for safety
    
    selected = []  # List of [m, r]
    
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        # Try to find a valid residue r for modulus m
        found = False
        for r in range(m):
            valid = True
            for (m_j, r_j) in selected:
                g = math.gcd(m, m_j)
                if r % g == r_j % g:
                    valid = False
                    break
            if valid:
                selected.append([m, r])
                found = True
                break
        
        if not found:
            # No valid residue for this m, skip it
            pass
    
    return selected
