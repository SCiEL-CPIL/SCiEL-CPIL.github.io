import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    # Moduli from 2 to 128
    moduli = list(range(2, 129))
    n_moduli = len(moduli)
    
    # Precompute GCD matrix
    # gcd_mat[i][j] = gcd(moduli[i], moduli[j])
    gcd_mat = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        mi = moduli[i]
        for j in range(n_moduli):
            gcd_mat[i][j] = math.gcd(mi, moduli[j])
            
    best_result = []
    best_size = 0
    
    # Pre-allocate indices list
    indices = list(range(n_moduli))
    
    while time.time() - start_time < deadline:
        random.shuffle(indices)
        current_list = [] # List of (residue, index)
        
        for idx in indices:
            m = moduli[idx]
            found = False
            # Try residues in random order
            # Optimization: Instead of shuffling a list of 128 items every time,
            # we can just pick random residues until we find one?
            # No, we need to ensure we don't miss a valid one if it's rare.
            # But with 128 residues and ~50 constraints, most residues are valid.
            # So random picking is very likely to succeed quickly.
            
            # Strategy: Try a few random residues first. If fail, do full shuffle?
            # Or just full shuffle, it's fast enough.
            
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                compatible = True
                for r_j, idx_j in current_list:
                    g = gcd_mat[idx][idx_j]
                    if (r - r_j) % g == 0:
                        compatible = False
                        break
                if compatible:
                    current_list.append((r, idx))
                    found = True
                    break
            
            # If not found, skip this modulus in this iteration
            
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_result = [[r, moduli[idx]] for r, idx in current_list]
            
            if best_size == n_moduli:
                break

    return best_result
