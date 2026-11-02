import numpy as np

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    """
    n, m, p = 2, 2, 16
    
    # Strassen coefficients for 2x2x2
    # U rows: [A00, A01, A10, A11]
    # V rows: [B00, B01, B10, B11]
    # W rows: [C00, C01, C10, C11]
    
    strassen_U = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ], dtype=float)
    
    strassen_V = np.array([
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ], dtype=float)
    
    strassen_W = np.array([
        [1, 0, 0, 1],   # M1: C00+1, C11+1
        [0, 0, 1, -1],  # M2: C10+1, C11-1
        [0, 1, 0, 1],   # M3: C01+1, C11+1
        [1, 0, 1, 0],   # M4: C00+1, C10+1
        [-1, 1, 0, 0],  # M5: C00-1, C01+1
        [0, 0, 0, 1],   # M6: C11+1
        [1, 0, 0, 0]    # M7: C00+1
    ], dtype=float)
    
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7  # 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for block in range(num_blocks):
        k0 = block * 2
        k1 = k0 + 1
        base = block * 7
        
        for s in range(7):
            col = base + s
            
            # U: same for all blocks
            U[:, col] = strassen_U[s]
            
            # V: map local B indices to global V indices
            V[k0, col] = strassen_V[s, 0]       # B00
            V[k1, col] = strassen_V[s, 1]       # B01
            V[p + k0, col] = strassen_V[s, 2]   # B10
            V[p + k1, col] = strassen_V[s, 3]   # B11
            
            # W: map local C indices to global W indices
            W[2*k0, col] = strassen_W[s, 0]     # C00
            W[2*k1, col] = strassen_W[s, 1]     # C01
            W[2*k0+1, col] = strassen_W[s, 2]   # C10
            W[2*k1+1, col] = strassen_W[s, 3]   # C11
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
