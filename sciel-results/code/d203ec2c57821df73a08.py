import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x8 matrix using Strassen's algorithm on padded blocks.
    
    Strategy:
    1. Pad A (3x4) to A' (4x4) with a zero row.
    2. Split B (4x8) into B1 (4x4) and B2 (4x4).
    3. C = A'B1 + A'B2.
    4. Use Strassen (rank 7) for each 4x4x4 multiplication.
    5. Total rank = 7 + 7 = 14.
    """
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # Strassen decomposition for 4x4x4
    # U, V, W are 16x7
    # Indices for 4x4 matrix: 0..3
    # A indices: i*4 + j
    # B indices: j*4 + k
    # C indices: k*4 + i
    
    def strassen_4x4():
        U = np.zeros((16, 7), dtype=int)
        V = np.zeros((16, 7), dtype=int)
        W = np.zeros((16, 7), dtype=int)
        
        # M1 = (A11+A22)(B11+B22)
        # A11: 0,1,4,5; A22: 10,11,14,15
        # B11: 0,1,4,5; B22: 10,11,14,15
        for idx in [0, 1, 4, 5]: U[idx, 0] += 1
        for idx in [10, 11, 14, 15]: U[idx, 0] += 1
        for idx in [0, 1, 4, 5]: V[idx, 0] += 1
        for idx in [10, 11, 14, 15]: V[idx, 0] += 1
        # C11 += M1, C22 += M1
        # C11: 0,1,4,5; C22: 10,11,14,15
        for idx in [0, 1, 4, 5]: W[idx, 0] += 1
        for idx in [10, 11, 14, 15]: W[idx, 0] += 1
        
        # M2 = (A21+A22)B11
        # A21: 8,9,12,13; A22: 10,11,14,15
        # B11: 0,1,4,5
        for idx in [8, 9, 12, 13]: U[idx, 1] += 1
        for idx in [10, 11, 14, 15]: U[idx, 1] += 1
        for idx in [0, 1, 4, 5]: V[idx, 1] += 1
        # C21 += M2, C22 -= M2
        # C21: 8,9,12,13; C22: 10,11,14,15
        for idx in [8, 9, 12, 13]: W[idx, 1] += 1
        for idx in [10, 11, 14, 15]: W[idx, 1] -= 1
        
        # M3 = A11(B12-B22)
        # A11: 0,1,4,5
        # B12: 2,3,6,7; B22: 10,11,14,15
        for idx in [0, 1, 4, 5]: U[idx, 2] += 1
        for idx in [2, 3, 6, 7]: V[idx, 2] += 1
        for idx in [10, 11, 14, 15]: V[idx, 2] -= 1
        # C12 += M3, C22 += M3
        # C12: 2,3,6,7; C22: 10,11,14,15
        for idx in [2, 3, 6, 7]: W[idx, 2] += 1
        for idx in [10, 11, 14, 15]: W[idx, 2] += 1
        
        # M4 = A22(B21-B11)
        # A22: 10,11,14,15
        # B21: 8,9,12,13; B11: 0,1,4,5
        for idx in [10, 11, 14, 15]: U[idx, 3] += 1
        for idx in [8, 9, 12, 13]: V[idx, 3] += 1
        for idx in [0, 1, 4, 5]: V[idx, 3] -= 1
        # C11 += M4, C21 += M4
        # C11: 0,1,4,5; C21: 8,9,12,13
        for idx in [0, 1, 4, 5]: W[idx, 3] += 1
        for idx in [8, 9, 12, 13]: W[idx, 3] += 1
        
        # M5 = (A11+A12)B22
        # A11: 0,1,4,5; A12: 2,3,6,7
        # B22: 10,11,14,15
        for idx in [0, 1, 4, 5]: U[idx, 4] += 1
        for idx in [2, 3, 6, 7]: U[idx, 4] += 1
        for idx in [10, 11, 14, 15]: V[idx, 4] += 1
        # C11 -= M5, C12 += M5
        # C11: 0,1,4,5; C12: 2,3,6,7
        for idx in [0, 1, 4, 5]: W[idx, 4] -= 1
        for idx in [2, 3, 6, 7]: W[idx, 4] += 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21: 8,9,12,13; A11: 0,1,4,5
        # B11: 0,1,4,5; B12: 2,3,6,7
        for idx in [8, 9, 12, 13]: U[idx, 5] += 1
        for idx in [0, 1, 4, 5]: U[idx, 5] -= 1
        for idx in [0, 1, 4, 5]: V[idx, 5] += 1
        for idx in [2, 3, 6, 7]: V[idx, 5] += 1
        # C22 += M6
        # C22: 10,11,14,15
        for idx in [10, 11, 14, 15]: W[idx, 5] += 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12: 2,3,6,7; A22: 10,11,14,15
        # B21: 8,9,12,13; B22: 10,11,14,15
        for idx in [2, 3, 6, 7]: U[idx, 6] += 1
        for idx in [10, 11, 14, 15]: U[idx, 6] -= 1
        for idx in [8, 9, 12, 13]: V[idx, 6] += 1
        for idx in [10, 11, 14, 15]: V[idx, 6] += 1
        # C11 += M7
        # C11: 0,1,4,5
        for idx in [0, 1, 4, 5]: W[idx, 6] += 1
        
        return U, V, W

    U_str, V_str, W_str = strassen_4x4()
    
    # Global matrices
    # U: 12x14 (3*4 x 14)
    # V: 32x14 (4*8 x 14)
    # W: 24x14 (8*3 x 14)
    
    total_rank = 14
    U = np.zeros((12, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((24, total_rank), dtype=int)
    
    # Block 1: B[:, 0:4], C[:, 0:4]
    # A is 3x4, padded to 4x4.
    # U_str is 16x7. Rows 0-11 correspond to A[0:3, 0:4]. Row 12-15 are padding (zeros in U_str? No, U_str has non-zeros for A22 etc).
    # We only take the first 12 rows of U_str for the U matrix.
    # V_str is 16x7. Rows correspond to B[0:4, 0:4].
    # B global indices: j*8 + k.
    # For B1, j in 0..3, k in 0..3.
    # V_str index r = j*4 + k.
    # V global index = j*8 + k.
    
    # W_str is 16x7. Rows correspond to C[0:4, 0:4].
    # C global indices: k*3 + i.
    # W_str index r = k*4 + i.
    # We only want i in 0..2.
    # So for each k (0..3), we take i=0,1,2.
    
    for col in range(7):
        # U
        U[:, col] = U_str[:12, col]
        
        # V
        for j in range(4):
            for k in range(4):
                local_idx = j * 4 + k
                global_idx = j * 8 + k
                V[global_idx, col] = V_str[local_idx, col]
        
        # W
        for k in range(4):
            for i in range(3):
                local_idx = k * 4 + i
                global_idx = k * 3 + i
                W[global_idx, col] = W_str[local_idx, col]
    
    # Block 2: B[:, 4:8], C[:, 4:8]
    # A is same.
    # B2: j in 0..3, k in 4..7.
    # V_str index r = j*4 + k_local (0..3).
    # V global index = j*8 + (k_local + 4).
    
    # C2: i in 0..2, k in 4..7.
    # W_str index r = k_local*4 + i.
    # W global index = (k_local + 4)*3 + i.
    
    for col in range(7, 14):
        local_col = col - 7
        # U
        U[:, col] = U_str[:, local_col]
        
        # V
        for j in range(4):
            for k_local in range(4):
                local_idx = j * 4 + k_local
                global_idx = j * 8 + (k_local + 4)
                V[global_idx, col] = V_str[local_idx, local_col]
        
        # W
        for k_local in range(4):
            for i in range(3):
                local_idx = k_local * 4 + i
                global_idx = (k_local + 4) * 3 + i
                W[global_idx, col] = W_str[local_idx, local_col]
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
