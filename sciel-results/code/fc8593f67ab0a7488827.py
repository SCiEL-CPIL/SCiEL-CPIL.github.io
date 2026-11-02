"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-98 decomposition for 4x4 by 4x8 matrix multiplication.
    
    Strategy: Use the known rank-49 decomposition for 4x4x4 (via recursive Strassen)
    and extend it to 4x4x8 by duplicating the structure for the two 4-column blocks.
    
    The tensor for C = A*B where A is 4x4 and B is 4x8 can be viewed as two independent
    4x4x4 multiplications (one for columns 0-3 of B, one for columns 4-7).
    
    We use the recursive Kronecker product of Strassen's 2x2x2 rank-7 decomposition.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Strassen 2x2x2 decomposition
    # U2, V2, W2 are 7x4 matrices representing the 7 rank-1 terms
    # Indexing: row-major flattening of 2x2 matrices: [00, 01, 10, 11]
    
    U2 = np.array([
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0]    # M7: C00
    ], dtype=int)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product of two Strassen decompositions
    # U4 will be 16x49, V4 will be 16x49, W4 will be 16x49
    
    # Vectorized construction using einsum
    # U2 is (7, 4) where 4 is (2,2) flattened.
    # We want U4 (16, 49) where 16 is (4,4) and 49 is (7,7).
    # U4[I,J, s,t] = U2[s, I//2, J//2] * U2[t, I%2, J%2]
    
    # Reshape U2 to (7, 2, 2)
    U2_3d = U2.reshape(7, 2, 2)
    V2_3d = V2.reshape(7, 2, 2)
    W2_3d = W2.reshape(7, 2, 2)
    
    # Compute U4
    # s, t from 0..6
    # I, J from 0..3
    # U4[I, J, s, t] = U2_3d[s, I//2, J//2] * U2_3d[t, I%2, J%2]
    
    # Use broadcasting
    # U2_3d[s, r, c] * U2_3d[t, i, j]
    # We need to sum over s, t? No, each (s,t) is a separate column in U4.
    
    # Create indices
    s_idx = np.arange(7)[:, None, None, None] # (7, 1, 1, 1)
    t_idx = np.arange(7)[None, :, None, None] # (1, 7, 1, 1)
    
    # I, J indices for 4x4
    I_idx = np.arange(4)[None, None, :, None] # (1, 1, 4, 1)
    J_idx = np.arange(4)[None, None, None, :] # (1, 1, 1, 4)
    
    # Block indices
    r_idx = I_idx // 2
    c_idx = J_idx // 2
    i_idx = I_idx % 2
    j_idx = J_idx % 2
    
    # U2_3d shape (7, 2, 2)
    # We want to extract U2_3d[s, r, c] and U2_3d[t, i, j]
    
    # Expand dims for broadcasting
    # U2_3d[s, r, c] -> (7, 1, 1, 1) x (1, 1, 4, 1) x (1, 1, 1, 4) ? No.
    
    # Let's just use the loop, it's fast enough for 49*16 operations.
    # But let's try to vectorize for style.
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    def idx2(i, j):
        return i * 2 + j
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U4 part
            for I in range(4):
                for J in range(4):
                    r = I // 2
                    c = J // 2
                    i_loc = I % 2
                    j_loc = J % 2
                    U4[I * 4 + J, col] = U2[s, idx2(r, c)] * U2[t, idx2(i_loc, j_loc)]
            
            # V4 part
            for J in range(4):
                for K in range(4):
                    r = J // 2
                    c = K // 2
                    j_loc = J % 2
                    k_loc = K % 2
                    V4[J * 4 + K, col] = V2[s, idx2(r, c)] * V2[t, idx2(j_loc, k_loc)]
            
            # W4 part
            for I in range(4):
                for K in range(4):
                    r = I // 2
                    c = K // 2
                    i_loc = I % 2
                    k_loc = K % 2
                    W4[I * 4 + K, col] = W2[s, idx2(r, c)] * W2[t, idx2(i_loc, k_loc)]
    
    # Now extend to 4x4x8: two copies of the 4x4x4 decomposition
    # Total rank = 49 + 49 = 98
    rank = 98
    
    # U: shape (n*m, rank) = (16, 98)
    # V: shape (m*p, rank) = (32, 98)
    # W: shape (p*n, rank) = (32, 98)
    
    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int)
    W = np.zeros((32, 98), dtype=int)
    
    # Left block
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[i * 4 + k, :]
    
    # Right block
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:98] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W4[i * 4 + k, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
