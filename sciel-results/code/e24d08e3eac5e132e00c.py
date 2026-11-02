import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = 120.0 # 120s limit, use 110s for safety
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ]) # This is identity, not Strassen. 
    # Correct Strassen U (for A blocks A11, A12, A21, A22)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U2 maps A blocks to M inputs
    U2 = np.array([
        [1, 0, 0, 1], # A11, A12, A21, A22 -> M1 needs A11, A22
        [0, 0, 1, 1], # M2 needs A21, A22
        [1, 0, 0, 0], # M3 needs A11
        [0, 0, 0, 1], # M4 needs A22
        [1, 1, 0, 0], # M5 needs A11, A12
        [-1, 0, 1, 0], # M6 needs A21, A11
        [0, 1, 0, -1]  # M7 needs A12, A22
    ]).T # Shape 4x7
    
    # V2 maps B blocks to M inputs
    V2 = np.array([
        [1, 0, 1, 0], # B11, B12, B21, B22 -> M1 needs B11, B22
        [1, 0, 0, 0], # M2 needs B11
        [0, 1, 0, -1], # M3 needs B12, B22
        [0, 0, 1, -1], # M4 needs B21, B11
        [0, 0, 0, 1], # M5 needs B22
        [1, 1, 0, 0], # M6 needs B11, B12
        [0, 0, 1, 1]  # M7 needs B21, B22
    ]).T # Shape 4x7
    
    # W2 maps M outputs to C blocks
    W2 = np.array([
        [1, 0, 0, 1], # C11 = M1 - M4 + M5 - M7 ? No.
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M2 - M6
        [1, -1, 0, 1], # C11 coeffs for M1..M7: 1, 0, 0, -1, 1, 0, -1
        [0, 0, 1, 0], # C12 coeffs: 0, 0, 1, 0, 1, 0, 0
        [0, 1, 0, 0], # C21 coeffs: 0, 1, 0, 1, 0, 0, 0
        [1, 0, -1, 1]  # C22 coeffs: 1, -1, 0, 1, 0, -1, 0 ? 
    ])
    # Let's use standard Strassen W
    W2 = np.array([
        [1, 0, 0, -1, 1, 0, -1], # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 0, 1, 0, -1, 0]  # C22
    ]).T # Shape 4x7

    # Construct 4x4x4 decomposition (Kronecker product)
    # U4 is 16x49, V4 is 16x49, W4 is 16x49
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # For 4x4x8, we split B (4x8) into B1 (4x4) and B2 (4x4)
    # C (4x8) is [C1, C2]
    # Rank = 49 + 49 = 98
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank)) # m*p = 4*8 = 32
    W = np.zeros((32, rank)) # p*n = 8*4 = 32
    
    # First 49 terms for C1 = A * B1
    U[:, :49] = U4
    # V4 expects B1 (4x4). B1 is first 4 columns of B.
    # V4 is 16x49. We need to map B1 (16 entries) to V4 inputs.
    # V4 is kron(V2, V2). 
    # We place V4 in the first 16 rows of V (corresponding to B1).
    V[:16, :49] = V4
    
    # W4 expects to output C1 (4x4). C1 is first 4 columns of C.
    # W4 is 16x49. We place W4 in the first 16 rows of W (corresponding to C1).
    W[:16, :49] = W4
    
    # Next 49 terms for C2 = A * B2
    U[:, 49:] = U4
    # B2 is last 4 columns of B. Indices 16..31 in V.
    V[16:, 49:] = V4
    
    # C2 is last 4 columns of C. Indices 16..31 in W.
    W[16:, 49:] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
