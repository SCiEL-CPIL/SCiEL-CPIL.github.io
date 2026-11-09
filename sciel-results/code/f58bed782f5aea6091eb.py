import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    moduli = list(range(20, 201))
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]

    sel_indices = []
    sel_residues = []

    for i in range(n):
        if time.time() > deadline:
            break
            
        m = moduli[i]
        valid_rs = []
        for r in range(m):
            is_valid = True
            for k_idx in range(len(sel_indices)):
                j = sel_indices[k_idx]
                m_j = moduli[j]
                r_j = sel_residues[k_idx]
                g = gcds[i][j]
                if r % g == r_j % g:
                    is_valid = False
                    break
            if is_valid:
                valid_rs.append(r)
        
        if not valid_rs:
            continue

        # Pick the smallest valid residue
        best_r = valid_rs[0]

        sel_indices.append(i)
        sel_residues.append(best_r)

    result = []
    for idx, r in zip(sel_indices, sel_residues):
        result.append([moduli[idx], r])
        
    return result
