import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 blocks * 7 Strassen factors = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # A indices: (i, j_local) where i in {0,1}, j_local in {0,1}
    # s in {0..6}
    # M1: (A00+A11)
    # M2: (A10+A11)
    # M3: A00
    # M4: A11
    # M5: (A00+A01)
    # M6: (A10-A00)
    # M7: (A01-A11)
    A_coeffs = [
        {(0,0): 1, (1,1): 1},  # s=0
        {(1,0): 1, (1,1): 1},  # s=1
        {(0,0): 1},            # s=2
        {(1,1): 1},            # s=3
        {(0,0): 1, (0,1): 1},  # s=4
        {(1,0): 1, (0,0): -1}, # s=5
        {(0,1): 1, (1,1): -1}  # s=6
    ]
    
    # Strassen coefficients for B (2x2)
    # B indices: (j_local, k_local)
    # M1: (B00+B11)
    # M2: B00
    # M3: (B10-B11)
    # M4: (B01-B00)
    # M5: B11
    # M6: (B00+B01)
    # M7: (B10+B11)
    B_coeffs = [
        {(0,0): 1, (1,1): 1},  # s=0
        {(0,0): 1},            # s=1
        {(1,0): 1, (1,1): -1}, # s=2
        {(0,1): 1, (0,0): -1}, # s=3
        {(1,1): 1},            # s=4
        {(0,0): 1, (0,1): 1},  # s=5
        {(1,0): 1, (1,1): 1}   # s=6
    ]
    
    # Strassen coefficients for C (2x2)
    # C indices: (i, k_local)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): 0, (0,0) : 1}, # Placeholder, will fill properly
    ]
    # Let's define C_coeffs as list of dicts for s=0..6
    # s=0 (M1): C00+=1, C11+=1
    # s=1 (M2): C10+=1, C11+=-1
    # s=2 (M3): C01+=1, C11+=1
    # s=3 (M4): C00+=1, C10+=1
    # s=4 (M5): C00+=-1, C01+=1
    # s=5 (M6): C11+=1
    # s=6 (M7): C00+=1
    C_coeffs = [
        {(0,0): 1, (1,1): 1},       # s=0
        {(1,0): 1, (1,1): -1},      # s=1
        {(0,1): 1, (1,1): 1},       # s=2
        {(0,0): 1, (1,0): 1},       # s=3
        {(0,0): -1, (0,1): 1},      # s=4
        {(1,1): 1},                 # s=5
        {(0,0): 1}                  # s=6
    ]

    for b in range(8):
        j_block = (b >> 2) & 1
        k_block = (b >> 1) & 1
        l = b & 1
        
        # J_b: {2*j_block, 2*j_block+1}
        # K_b: {4*k_block + 2*l, 4*k_block + 2*l + 1}
        
        for s in range(7):
            c = b * 7 + s
            
            # Fill U
            for i in range(2):
                for j_local in range(2):
                    j = 2 * j_block + j_local
                    u_row = i * m + j
                    val = A_coeffs[s].get((i, j_local), 0)
                    if val != 0:
                        U[u_row][c] = val
                        
            # Fill V
            for j_local in range(2):
                j = 2 * j_block + j_local
                for k_local in range(2):
                    k = 4 * k_block + 2 * l + k_local
                    v_row = j * p + k
                    val = B_coeffs[s].get((j_local, k_local), 0)
                    if val != 0:
                        V[v_row][c] = val
                        
            # Fill W
            for i in range(2):
                for k_local in range(2):
                    k = 4 * k_block + 2 * l + k_local
                    w_row = k * n + i
                    val = C_coeffs[s].get((i, k_local), 0)
                    if val != 0:
                        W[w_row][c] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
