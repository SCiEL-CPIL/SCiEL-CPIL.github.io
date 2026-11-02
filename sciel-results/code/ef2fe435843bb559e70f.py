"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # Strategy:
    # Decompose the 2x15 matrix B into seven 2x2 blocks and one 2x1 block.
    # Use Strassen's algorithm (rank 7) for each 2x2 block.
    # Use schoolbook multiplication (rank 4) for the 2x1 block.
    # Total rank = 7 * 7 + 4 = 53.
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    # Strassen coefficients for M_{2,2,2}
    # U: 4x7 (indices i*2+j), V: 4x7 (indices j*2+k), W: 4x7 (indices k*2+i)
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0], # i=0,j=0 (A00)
        [0, 0, 0, 0, 1, 0, 1],  # i=0,j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0 (A10)
        [1, 1, 0, 1, 0, 0, -1]  # i=1,j=1 (A11)
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0], # j=0,k=0 (B00)
        [0, 0, 1, 0, 0, 1, 0],  # j=0,k=1 (B01)
        [0, 0, 0, 1, 0, 0, 1],  # j=1,k=0 (B10)
        [1, 0, -1, 0, 1, 0, 1]  # j=1,k=1 (B11)
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # k=0,i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],  # k=0,i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],  # k=1,i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]  # k=1,i=1 (C11)
    ]
    
    # Schoolbook coefficients for M_{2,2,1}
    # U: 4x4, V: 2x4, W: 2x4
    U_school = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    
    V_school = [
        [1, 0, 1, 0], # j=0 (B0)
        [0, 1, 0, 1]  # j=1 (B1)
    ]
    
    W_school = [
        [1, 1, 0, 0], # i=0 (C0)
        [0, 0, 1, 1]  # i=1 (C1)
    ]
    
    # Fill global matrices
    col_offset = 0
    for block in range(7):
        # Block uses columns k0 = 2*block and k1 = 2*block + 1 of B
        k0 = 2 * block
        k1 = 2 * block + 1
        
        for r in range(7):
            c = col_offset + r
            # Fill U (independent of block, same pattern)
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            
            # Fill V
            # V index: j * p + k
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            
            # Fill W
            # W index: k * n + i
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        
        col_offset += 7
    
    # Last block: M_{2,2,1} at k=14
    k_last = 14
    for r in range(4):
        c = col_offset + r
        # Fill U
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        
        # Fill V
        # V index: j * p + k
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        
        # Fill W
        # W index: k * n + i
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
