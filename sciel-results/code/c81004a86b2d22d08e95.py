import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Moduli range [20, 200]
    # We want to maximize sum(1/m), so we prioritize smaller m.
    # Constraint: For any pair m_i, m_j, gcd(m_i, m_j) must be > 1 
    # (otherwise they always intersect).
    # And residues r_i, r_j must satisfy r_i != r_j mod gcd(m_i, m_j).
    
    selected = []  # List of (m, r)
    
    # Iterate through potential moduli from smallest to largest
    for m in range(20, 201):
        if time.time() > deadline:
            break
            
        # Check if m can be added to the current set
        # 1. gcd(m, existing_m) must be > 1 for all existing_m
        compatible = True
        for existing_m, existing_r in selected:
            if math.gcd(m, existing_m) == 1:
                compatible = False
                break
        
        if not compatible:
            continue
            
        # 2. Find a residue r for m such that for all existing (existing_m, existing_r):
        #    r != existing_r mod gcd(m, existing_m)
        valid_residue = -1
        for r in range(m):
            conflict = False
            for existing_m, existing_r in selected:
                g = math.gcd(m, existing_m)
                if r % g == existing_r % g:
                    conflict = True
                    break
            if not conflict:
                valid_residue = r
                break
                
        if valid_residue != -1:
            selected.append((m, valid_residue))
            
    # Convert to required format [[m, r], ...]
    return [[m, r] for m, r in selected]
