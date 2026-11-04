"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition for 2x3x3
    rank = 17
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to set column
    def set_col(col, u_indices, v_indices, w_indices):
        for idx in u_indices:
            U[idx, col] = 1
        for idx in v_indices:
            V[idx, col] = 1
        for idx, val in w_indices:
            W[idx, col] = val

    # Indices mapping:
    # U: i*m + j (i:0-1, j:0-2) -> 0:00, 1:01, 2:02, 3:10, 4:11, 5:12
    # V: j*p + k (j:0-2, k:0-2) -> 0:00, 1:01, 2:02, 3:10, 4:11, 5:12, 6:20, 7:21, 8:22
    # W: k*n + i (k:0-2, i:0-1) -> 0:00, 1:10, 2:01, 3:11, 4:02, 5:12
    
    # Block 1: Strassen 2x2x2 for j=0,1 and k=0,1
    # M1 = (A00 + A11)(B00 + B11)
    set_col(0, [0, 4], [0, 4], [(0, 1), (3, 1)])
    # M2 = (A10 + A11)B00
    set_col(1, [3, 4], [0], [(1, 1), (3, -1)])
    # M3 = A00(B01 - B11)
    set_col(2, [0], [1, 4], [(2, 1), (3, 1)])
    # V[4] is -1 in M3? No, set_col adds 1. Need to handle signs.
    # Let's rewrite set_col to take values or handle signs in indices.
    # Easier to just assign directly.
    
    # Reset and do manual assignment for clarity and correctness
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Col 0: M1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Col 1: M2
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Col 2: M3
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Col 3: M4
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Col 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Col 5: M6
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Block 2: 2x2x1 for j=0,1 and k=2
    # Col 7: A00 B02
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1
    
    # Col 8: A01 B12
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1
    
    # Col 9: A10 B02
    U[3, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # Col 10: A11 B12
    U[4, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    # Block 3: 2x1x3 for j=2
    # Col 11: A02 B20
    U[2, 11] = 1
    V[6, 11] = 1
    W[0, 11] = 1
    
    # Col 12: A02 B21
    U[2, 12] = 1
    V[7, 12] = 1
    W[2, 12] = 1
    
    # Col 13: A02 B22
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # Col 14: A12 B20
    U[5, 14] = 1
    V[6, 14] = 1
    W[1, 14] = 1
    
    # Col 15: A12 B21
    U[5, 15] = 1
    V[7, 15] = 1
    W[3, 15] = 1
    
    # Col 16: A12 B22
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
