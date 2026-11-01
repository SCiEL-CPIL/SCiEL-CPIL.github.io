"""Tensor decomposition for 2x2 by 2x16 matrix multiplication using Strassen blocks."""
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    num_blocks = p // 2   # 8 blocks
    rank = num_blocks * 7 # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # alpha[i][j] for A
    # beta[j][k] for B (local k)
    # gamma[i][k] for C (local k)
    
    # s=0: M1
    # A: A00=1, A11=1
    # B: B00=1, B11=1
    # C: C00=1, C11=1
    strassen_coeffs = [
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        # s=1: M2
        # A: A10=1, A11=1
        # B: B00=1
        # C: C10=1, C11=-1
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        # s=2: M3
        # A: A00=1
        # B: B10=1, B11=-1
        # C: C01=1, C11=1
        {
            'A': {(0,0): 1},
            'B': {(1,0): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        # s=3: M4
        # A: A11=1
        # B: B01=1, B00=-1
        # C: C00=1, C10=1
        {
            'A': {(1,1): 1},
            'B': {(0,1): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        # s=4: M5
        # A: A00=1, A01=1
        # B: B11=1
        # C: C00=-1, C01=1
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        # s=5: M6
        # A: A10=1, A00=-1
        # B: B00=1, B01=1
        # C: C11=1
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        # s=6: M7
        # A: A01=1, A11=-1
        # B: B10=1, B11=1
        # C: C00=1
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]
    
    for block in range(num_blocks):
        k0 = block * 2
        k1 = k0 + 1
        base = block * 7
        
        for s in range(7):
            coeffs = strassen_coeffs[s]
            r = base + s
            
            # Fill U
            for (i, j), val in coeffs['A'].items():
                U[i * m + j][r] = val
                
            # Fill V
            for (j, k_loc), val in coeffs['B'].items():
                k_global = k0 + k_loc
                V[j * p + k_global][r] = val
                
            # Fill W
            for (i, k_loc), val in coeffs['C'].items():
                k_global = k0 + k_loc
                W[k_global * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
