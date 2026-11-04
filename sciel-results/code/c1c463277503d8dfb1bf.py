"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition of 5x5 into 4x4 and 1x1 blocks.
    # A = [[A44, A41], [A14, A11]]
    # B = [[B44, B41], [B14, B11]]
    # C = [[C44, C41], [C14, C11]]
    
    # C44 = A44 B44 + A41 B14
    # C41 = A44 B41 + A41 B11
    # C14 = A14 B44 + A11 B14
    # C11 = A14 B41 + A11 B11
    
    # Ranks:
    # A44 B44: 4x4 schoolbook rank 16
    # A41 B14: 4x1 * 1x4 outer product rank 1
    # A44 B41: 4x4 * 4x1 mat-vec rank 16
    # A41 B11: 4x1 * 1x1 rank 1
    # A14 B44: 1x4 * 4x4 vec-mat rank 16
    # A11 B14: 1x1 * 1x4 rank 1
    # A14 B41: 1x4 * 4x1 inner product rank 4
    # A11 B11: 1x1 * 1x1 rank 1
    
    total_rank = 16 + 1 + 16 + 1 + 16 + 1 + 4 + 1
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col = 0
    
    # 1. A44 B44 (16 terms)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5 + j, col] = 1
                V[j*5 + k, col] = 1
                W[k*5 + i, col] = 1
                col += 1
                
    # 2. A41 B14 (1 term)
    # A41: i=0..3, j=4.
    # B14: i=4, j=0..3.
    # C44: i=0..3, k=0..3.
    for i in range(4):
        U[i*5 + 4, col] = 1
    for k in range(4):
        V[4*5 + k, col] = 1
    for i in range(4):
        for k in range(4):
            W[k*5 + i, col] = 1
    col += 1
    
    # 3. A44 B41 (16 terms)
    # C41: i=0..3, k=0.
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 0, col] = 1
            W[0*5 + i, col] = 1
            col += 1
            
    # 4. A41 B11 (1 term)
    for i in range(4):
        U[i*5 + 4, col] = 1
    V[4*5 + 4, col] = 1
    for i in range(4):
        W[0*5 + i, col] = 1
    col += 1
    
    # 5. A14 B44 (16 terms)
    # C14: i=4, k=0..3.
    for j in range(4):
        for k in range(4):
            U[4*5 + j, col] = 1
            V[j*5 + k, col] = 1
            W[k*5 + 4, col] = 1
            col += 1
            
    # 6. A11 B14 (1 term)
    U[4*5 + 4, col] = 1
    for k in range(4):
        V[4*5 + k, col] = 1
        W[k*5 + 4, col] = 1
    col += 1
    
    # 7. A14 B41 (4 terms)
    # C11: i=4, k=0.
    for j in range(4):
        U[4*5 + j, col] = 1
        V[j*5 + 0, col] = 1
        W[0*5 + 4, col] = 1
        col += 1
        
    # 8. A11 B11 (1 term)
    U[4*5 + 4, col] = 1
    V[4*5 + 4, col] = 1
    W[0*5 + 4, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
