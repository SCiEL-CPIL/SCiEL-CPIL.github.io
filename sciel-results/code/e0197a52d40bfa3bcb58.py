"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Rank-49 exact tensor decomposition for 2x2 by 2x14 matrix multiplication.
    Uses Strassen's algorithm on each 2x2 block of the 2x14 matrix.
    """
    n, m, p = DIMENSIONS
    # p = 14, so 7 blocks of size 2x2
    n_blocks = p // 2
    rank = 7 * n_blocks  # 49
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # Strassen linear forms for left matrix (indices: 0=A00, 1=A01, 2=A10, 3=A11)
    # u1 = A00 + A11, u2 = A10 + A11, u3 = A00, u4 = A11
    # u5 = A00 + A01, u6 = A10 - A00, u7 = A01 - A11
    strassen_u = [
        [1, 0, 0, 1],   # u1
        [0, 0, 1, 1],   # u2
        [1, 0, 0, 0],   # u3
        [0, 0, 0, 1],   # u4
        [1, 1, 0, 0],   # u5
        [-1, 0, 1, 0],  # u6
        [0, 1, 0, -1]   # u7
    ]
    
    # Strassen linear forms for right block (indices: 0=b00, 1=b01, 2=b10, 3=b11)
    # v1 = b00 + b11, v2 = b00, v3 = b01 - b11, v4 = b10 - b00
    # v5 = b11, v6 = b00 + b01, v7 = b10 + b11
    strassen_v = [
        [1, 0, 0, 1],   # v1
        [1, 0, 0, 0],   # v2
        [0, 1, 0, -1],  # v3
        [-1, 0, 1, 0],  # v4
        [0, 0, 0, 1],   # v5
        [1, 1, 0, 0],   # v6
        [0, 0, 1, 1]    # v7
    ]
    
    # Reconstruction coefficients for C = [[C00, C01], [C10, C11]]
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # Output indices: 0=C00, 1=C01, 2=C10, 3=C11
    strassen_w = [
        [1, 0, 0, 0],   # P1 -> C00
        [0, 0, 1, 0],   # P2 -> C10
        [0, 1, 0, 0],   # P3 -> C01
        [1, 0, 1, 0],   # P4 -> C00, C10
        [-1, 1, 0, 0],  # P5 -> -C00, C01
        [0, 0, 0, 1],   # P6 -> C11
        [1, 0, 0, 0]    # P7 -> C00
    ]
    
    for block in range(n_blocks):
        for s in range(7):
            r = block * 7 + s
            
            # Set U: apply strassen_u[s] to the 4 elements of A
            for i in range(4):
                U[i][r] = strassen_u[s][i]
            
            # Set V: apply strassen_v[s] to the 2x2 block B[:, 2*block:2*block+2]
            # B indices: row 0 -> cols 2*block, 2*block+1 => indices 0, 1 in local
            # B indices: row 1 -> cols 2*block, 2*block+1 => indices 2, 3 in local
            # Global V index for row j, col k is j * p + k
            # Local col 0 -> global col 2*block, local col 1 -> global col 2*block+1
            for local_col in range(4):
                if strassen_v[s][local_col] != 0:
                    if local_col < 2:
                        row = 0
                        col = 2 * block + local_col
                    else:
                        row = 1
                        col = 2 * block + (local_col - 2)
                    V[row * p + col][r] = strassen_v[s][local_col]
            
            # Set W: distribute to C[:, 2*block:2*block+2]
            # C indices: row 0 -> cols 2*block, 2*block+1 => indices 0, 1 in local
            # C indices: row 1 -> cols 2*block, 2*block+1 => indices 2, 3 in local
            # Global W index for row k, col i is k * n + i
            # Wait, W has shape (p*n, rank). Index is k*n + i where k is col of C (0..p-1), i is row of C (0..n-1)
            # C row 0, col 2*block -> W[2*block * n + 0]
            # C row 0, col 2*block+1 -> W[2*block * n + 1]
            # C row 1, col 2*block -> W[2*block * n + 2]
            # C row 1, col 2*block+1 -> W[2*block * n + 3]
            for local_idx in range(4):
                if strassen_w[s][local_idx] != 0:
                    if local_idx < 2:
                        row_c = 0
                        col_c = 2 * block + local_idx
                    else:
                        row_c = 1
                        col_c = 2 * block + (local_idx - 2)
                    W[col_c * n + row_c][r] = strassen_w[s][local_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
