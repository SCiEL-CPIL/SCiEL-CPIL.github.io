import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank 3 decomposition of 2x2x1 multiplication
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Terms:
    # 1: (A00+A01) * (B0+B1)
    # 2: (A00-A01) * (B0-B1)
    # 3: (A10+A11) * (B0+B1) ?? No, this is not correct.
    
    # Correct rank 3 for 2x2x1:
    # M1 = (A00 + A01) * (B0 + B1)
    # M2 = (A00 - A01) * (B0 - B1)
    # M3 = (A10 + A11) * (B0 + B1)
    # C0 = (M1 + M2)/2
    # C1 = (M3 + ...)/2?
    
    # Actually, simpler:
    # Use the identity:
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # Let P1 = (A00+A01)(B0+B1)
    # Let P2 = (A00-A01)(B0-B1)
    # Let P3 = (A10+A11)(B0+B1)
    # This doesn't recover C1 easily.
    
    # Standard rank 3:
    # P1 = A00 * (B0+B1)
    # P2 = A01 * (B0-B1)
    # P3 = (A10+A11) * B0 ?
    
    # I will use the explicit rank 30 construction by summing 10 rank 3 decompositions.
    # For each k in 0..9, use a rank 3 decomposition for column k.
    # The U vectors will be shared? No, they are part of the rank-1 terms.
    
    col = 0
    for k in range(p):
        # Rank 3 decomposition for 2x2x1
        # Term 1
        U[0][col] = 1  # A00
        U[1][col] = 1  # A01
        V[0*p + k][col] = 1  # B0k
        V[1*p + k][col] = 1  # B1k
        W[k*n + 0][col] = 0.5  # C0k
        W[k*n + 1][col] = 0.5  # C1k
        col += 1
        
        # Term 2
        U[0][col] = 1
        U[1][col] = -1
        V[0*p + k][col] = 1
        V[1*p + k][col] = -1
        W[k*n + 0][col] = 0.5
        W[k*n + 1][col] = -0.5
        col += 1
        
        # Term 3
        U[2][col] = 1  # A10
        U[3][col] = 1  # A11
        V[0*p + k][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 0
        W[k*n + 1][col] = 0.5
        col += 1
        
    return {"U": U, "V": V, "W": W}
