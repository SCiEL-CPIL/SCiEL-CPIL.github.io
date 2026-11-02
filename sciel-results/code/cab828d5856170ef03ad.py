import numpy as np

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2
    # s: 0..6
    # alpha[s]: dict (i, j) -> val
    # beta[s]: dict (j, k) -> val
    # gamma[s]: dict (k, i) -> val
    
    # Corrected Strassen Coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    alpha = [
        {(0,0): 1, (1,1): 1},   # M1
        {(1,0): 1, (1,1): 1},   # M2
        {(0,0): 1},             # M3
        {(1,1): 1},             # M4
        {(0,0): 1, (0,1): 1},   # M5
        {(1,0): 1, (0,0): -1},  # M6
        {(0,1): 1, (1,1): -1}   # M7
    ]
    
    beta = [
        {(0,0): 1, (1,1): 1},   # M1
        {(0,0): 1},             # M2
        {(0,1): 1, (1,1): -1},  # M3
        {(1,0): 1, (0,0): -1},  # M4
        {(1,1): 1},             # M5
        {(0,0): 1, (0,1): 1},   # M6
        {(1,0): 1, (1,1): 1}    # M7
    ]
    
    # Gamma maps to output C[i, k]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # gamma[s] is a dict (k_local, i) -> val
    gamma = [
        {(0,0): 1, (1,1): 1},   # M1: C00 += 1, C11 += 1
        {(0,1): 1, (1,1): -1},  # M2: C10 += 1, C11 -= 1
        {(1,0): 1, (1,1): 1},   # M3: C01 += 1, C11 += 1
        {(0,0): 1, (0,1): 1},   # M4: C00 += 1, C10 += 1
        {(0,0): -1, (1,0): 1},  # M5: C00 -= 1, C01 += 1
        {(1,1): 1},             # M6: C11 += 1
        {(0,0): 1}              # M7: C00 += 1
    ]
    
    for b in range(3):
        for s in range(7):
            r = b * 7 + s
            
            # Fill U (same for all blocks)
            for (i, j), val in alpha[s].items():
                U[i * m + j, r] = val
                
            # Fill V (shifted by block b)
            for (j, k_local), val in beta[s].items():
                k_global = b * 2 + k_local
                V[j * p + k_global, r] = val
                
            # Fill W (shifted by block b)
            for (k_local, i), val in gamma[s].items():
                k_global = b * 2 + k_local
                W[k_global * n + i, r] = val
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
