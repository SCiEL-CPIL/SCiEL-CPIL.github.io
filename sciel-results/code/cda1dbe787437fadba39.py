import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for 2x2 by 2x8 (split into four 2x2 by 2x2)
    # A cols 0,1. B rows 0,1.
    # B cols split into 4 blocks of 2.
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Term 0: M1 = (A00 + A11) * (B0k0 + B1k1)
        col = 7 * b + 0
        U[0 * m + 0][col] += 1  # A00
        U[1 * m + 1][col] += 1  # A11
        V[0 * p + k0][col] += 1 # B0k0
        V[1 * p + k1][col] += 1 # B1k1
        
        W[k0 * n + 0][col] += 1 # C0k0
        W[k1 * n + 1][col] += 1 # C1k1
        # Check: C0k0 gets +1, C1k1 gets +1.
        # M1 contributes to C00 and C11 in 2x2.
        # C0k0 is (0,0) of block. C1k1 is (1,1) of block.
        
        # Term 1: M2 = (A10 + A11) * B0k0
        col = 7 * b + 1
        U[1 * m + 0][col] += 1  # A10
        U[1 * m + 1][col] += 1  # A11
        V[0 * p + k0][col] += 1 # B0k0
        
        W[k0 * n + 1][col] += 1 # C1k0
        # M2 contributes to C10 in 2x2.
        
        # Term 2: M3 = A00 * B1k0
        col = 7 * b + 2
        U[0 * m + 0][col] += 1  # A00
        V[1 * p + k0][col] += 1 # B1k0
        
        W[k1 * n + 0][col] += 1 # C0k1
        W[k0 * n + 1][col] += 1 # C1k0
        # M3 contributes to C01 and C10?
        # Formula: C01 = M3 + M5. C10 = M2 + M4.
        # Wait, my previous formula check:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M3 contributes to C01 (k1, row 0) and C11 (k1, row 1)?
        # No, C11 = ... + M3.
        # So M3 contributes to C0k1 and C1k1.
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        
        # Term 3: M4 = A11 * B1k1
        col = 7 * b + 3
        U[1 * m + 1][col] += 1  # A11
        V[1 * p + k1][col] += 1 # B1k1
        
        W[k0 * n + 0][col] += 1 # C0k0
        W[k0 * n + 1][col] += 1 # C1k0
        W[k1 * n + 1][col] += 1 # C1k1
        # M4 contributes to C00, C10, C11.
        
        # Term 4: M5 = A00 * (B0k1 + B1k1)
        col = 7 * b + 4
        U[0 * m + 0][col] += 1  # A00
        V[0 * p + k1][col] += 1 # B0k1
        V[1 * p + k1][col] += 1 # B1k1
        
        W[k0 * n + 0][col] += -1 # C0k0
        W[k1 * n + 0][col] += 1  # C0k1
        # M5 contributes to -C00 and +C01.
        
        # Term 5: M6 = (A01 - A10) * (B1k0 + B1k1)
        col = 7 * b + 5
        U[0 * m + 1][col] += 1  # A01
        U[1 * m + 0][col] += -1 # A10
        V[1 * p + k0][col] += 1 # B1k0
        V[1 * p + k1][col] += 1 # B1k1
        
        W[k1 * n + 1][col] += 1 # C1k1
        # M6 contributes to C11.
        
        # Term 6: M7 = (A00 + A01) * B0k1
        col = 7 * b + 6
        U[0 * m + 0][col] += 1  # A00
        U[0 * m + 1][col] += 1  # A01
        V[0 * p + k1][col] += 1 # B0k1
        
        W[k0 * n + 0][col] += 1 # C0k0
        # M7 contributes to C00.

    # Part 2: Outer product for 2x1 by 1x8
    # A col 2. B row 2.
    for i in range(2):
        for k in range(8):
            col = 28 + i * 8 + k
            U[i * m + 2][col] += 1  # Ai2
            V[2 * p + k][col] += 1  # B2k
            W[k * n + i][col] += 1  # Cik

    return {"U": U, "V": V, "W": W}
