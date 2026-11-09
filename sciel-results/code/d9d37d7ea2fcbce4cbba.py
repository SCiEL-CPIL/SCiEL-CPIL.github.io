"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Uses 4 independent Strassen rank-7 decompositions, total rank 28.
    
    U: (n*m, rank) = (8, 28)  -- U[i*m + j, c]
    V: (m*p, rank) = (16, 28) -- V[j*p + k, c]  
    W: (p*n, rank) = (8, 28)  -- W[k*n + i, c]
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Each term: U entries are (local_i, local_j, coeff)
    #            V entries are (local_j, local_k, coeff)
    #            W entries are (local_k, local_i, coeff)
    strassen_terms = [
        # M1 = (a+d)(e+h)
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        # M2 = (c+d)*e
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(0, 1, 1), (1, 1, -1)]
        },
        # M3 = a*(f-h)
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(1, 0, 1), (1, 1, 1)]
        },
        # M4 = d*(g-e)
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (0, 1, 1)]
        },
        # M5 = (a+b)*h
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (1, 0, 1)]
        },
        # M6 = (c-a)*(e+f)
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        # M7 = (b-d)*(g+h)
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    # Four 2x2 blocks:
    # Block (j_off, k_off): A rows i=0,1; cols j=j_off, j_off+1
    #                        B rows j=j_off, j_off+1; cols k=k_off, k_off+1
    #                        C rows i=0,1; cols k=k_off, k_off+1
    blocks = [
        (0, 0),  # A[:,0:2] * B[0:2, 0:2] -> C[:, 0:2]
        (0, 2),  # A[:,0:2] * B[0:2, 2:4] -> C[:, 2:4]
        (2, 0),  # A[:,2:4] * B[2:4, 0:2] -> C[:, 0:2]
        (2, 2),  # A[:,2:4] * B[2:4, 2:4] -> C[:, 2:4]
    ]
    
    col = 0
    for j_off, k_off in blocks:
        for term in strassen_terms:
            for (li, lj, coeff) in term['U']:
                global_i = li
                global_j = lj + j_off
                U[global_i * m + global_j][col] += coeff
            
            for (lj, lk, coeff) in term['V']:
                global_j = lj + j_off
                global_k = lk + k_off
                V[global_j * p + global_k][col] += coeff
            
            for (lk, li, coeff) in term['W']:
                global_k = lk + k_off
                global_i = li
                W[global_k * n + global_i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
