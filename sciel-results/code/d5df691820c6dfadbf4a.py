import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _build_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    U[i, r] corresponds to A[i//2, i%2]
    V[j, r] corresponds to B[j//2, j%2]
    W[k, r] corresponds to C[k//2, k%2]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # Indices for A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices for B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Indices for C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    # Strassen variables:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1

    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1

    return U, V, W

def _build_strassen_2x4():
    """
    Builds a rank-28 decomposition for 2x4 matrix multiplication.
    Splits B (4x4) into two 4x2 blocks? No, let's split B into two 2x2 blocks vertically?
    Actually, standard Strassen applies to square matrices.
    For 2x4 * 4x4, we can view it as two 2x2 * 2x2 multiplications?
    B is 4x4. Split B into B1 (rows 0,1) and B2 (rows 2,3)?
    Then C = A * B = A * [B1; B2] = A*B1 + A*B2.
    A is 2x4. Split A into A1 (cols 0,1) and A2 (cols 2,3).
    Then A*B1 = A1*B11 + A2*B21? No.
    A = [A1 A2] where A1 is 2x2, A2 is 2x2.
    B = [B1; B2] where B1 is 4x2? No, B is 4x4.
    Let's partition indices.
    A has rows 0,1 and cols 0,1,2,3.
    B has rows 0,1,2,3 and cols 0,1,2,3.
    
    Let's just use the block structure:
    C (2x4) = A (2x4) * B (4x4).
    Split A into A_left (2x2, cols 0,1) and A_right (2x2, cols 2,3).
    Split B into B_top (4x2, rows 0,1? No, B rows match A cols)
    B rows 0,1 correspond to A cols 0,1.
    B rows 2,3 correspond to A cols 2,3.
    
    Let B1 be rows 0,1 of B (4x2? No, B is 4x4, so B1 is 2x4 submatrix? No).
    B is 4x4.
    Rows of B: 0,1,2,3.
    Cols of B: 0,1,2,3.
    
    C[0..1, 0..3]
    C[:, 0..1] depends on A[:, 0..1] * B[0..1, 0..1] + A[:, 2..3] * B[2..3, 0..1]
    C[:, 2..3] depends on A[:, 0..1] * B[0..1, 2..3] + A[:, 2..3] * B[2..3, 2..3]
    
    So we have 4 sub-multiplications of 2x2 matrices:
    1. A_L * B_TL -> contributes to C_TL
    2. A_R * B_BL -> contributes to C_TL
    3. A_L * B_TR -> contributes to C_TR
    4. A_R * B_BR -> contributes to C_TR
    
    Each 2x2 * 2x2 has rank 7.
    Total rank = 4 * 7 = 28.
    
    This seems correct and efficient.
    """
    U_s, V_s, W_s = _build_strassen_2x2()
    
    # Target: A (2x4), B (4x4), C (2x4)
    # U shape: (8, 28)  # A has 8 elements
    # V shape: (16, 28) # B has 16 elements
    # W shape: (8, 28)  # C has 8 elements
    
    U = np.zeros((8, 28))
    V = np.zeros((16, 28))
    W = np.zeros((8, 28))
    
    # Block 1: A_L (cols 0,1) * B_TL (rows 0,1, cols 0,1) -> C_TL (rows 0,1, cols 0,1)
    # A_L indices in A: 0,1 (row 0), 4,5 (row 1)?
    # A is 2x4. Index a = i*4 + j.
    # A_L: i in 0..1, j in 0..1. Indices: 0,1,4,5.
    # B_TL: i in 0..1, j in 0..1. Indices: 0,1,4,5.
    # C_TL: i in 0..1, j in 0..1. Indices: 0,1,4,5.
    
    # Mapping for Block 1
    # U_s index u_idx corresponds to A element at (u_idx//2, u_idx%2)
    # We need to map A_L (2x2) to U_s (4x7).
    # A_L element (i, j) is at A index i*4 + j.
    # U_s element (i, j) is at U_s index i*2 + j.
    # So U[i*4+j, 0..7] = U_s[i*2+j, 0..7]
    
    for u_idx in range(4):
        i, j = divmod(u_idx, 2)
        A_idx = i * 4 + j
        U[A_idx, 0:7] = U_s[u_idx, 0:7]
        
    for v_idx in range(4):
        i, j = divmod(v_idx, 2)
        B_idx = i * 4 + j
        V[B_idx, 0:7] = V_s[v_idx, 0:7]
        
    for w_idx in range(4):
        i, j = divmod(w_idx, 2)
        C_idx = i * 4 + j
        W[C_idx, 0:7] = W_s[w_idx, 0:7]
        
    # Block 2: A_R (cols 2,3) * B_BL (rows 2,3, cols 0,1) -> C_TL
    # A_R: i in 0..1, j in 2..3. Indices: 2,3,6,7.
    # B_BL: i in 2..3, j in 0..1. Indices: 8,9,12,13.
    # C_TL: i in 0..1, j in 0..1. Indices: 0,1,4,5.
    
    for u_idx in range(4):
        i, j = divmod(u_idx, 2)
        A_idx = i * 4 + (j + 2)
        U[A_idx, 7:14] = U_s[u_idx, 0:7]
        
    for v_idx in range(4):
        i, j = divmod(v_idx, 2)
        B_idx = (i + 2) * 4 + j
        V[B_idx, 7:14] = V_s[v_idx, 0:7]
        
    for w_idx in range(4):
        i, j = divmod(w_idx, 2)
        C_idx = i * 4 + j
        W[C_idx, 7:14] += W_s[w_idx, 0:7]
        
    # Block 3: A_L (cols 0,1) * B_TR (rows 0,1, cols 2,3) -> C_TR
    # A_L: Indices 0,1,4,5.
    # B_TR: i in 0..1, j in 2..3. Indices: 2,3,6,7.
    # C_TR: i in 0..1, j in 2..3. Indices: 2,3,6,7.
    
    for u_idx in range(4):
        i, j = divmod(u_idx, 2)
        A_idx = i * 4 + j
        U[A_idx, 14:21] = U_s[u_idx, 0:7]
        
    for v_idx in range(4):
        i, j = divmod(v_idx, 2)
        B_idx = i * 4 + (j + 2)
        V[B_idx, 14:21] = V_s[v_idx, 0:7]
        
    for w_idx in range(4):
        i, j = divmod(w_idx, 2)
        C_idx = i * 4 + (j + 2)
        W[C_idx, 14:21] = W_s[w_idx, 0:7]
        
    # Block 4: A_R (cols 2,3) * B_BR (rows 2,3, cols 2,3) -> C_TR
    # A_R: Indices 2,3,6,7.
    # B_BR: i in 2..3, j in 2..3. Indices: 10,11,14,15.
    # C_TR: Indices 2,3,6,7.
    
    for u_idx in range(4):
        i, j = divmod(u_idx, 2)
        A_idx = i * 4 + (j + 2)
        U[A_idx, 21:28] = U_s[u_idx, 0:7]
        
    for v_idx in range(4):
        i, j = divmod(v_idx, 2)
        B_idx = (i + 2) * 4 + (j + 2)
        V[B_idx, 21:28] = V_s[v_idx, 0:7]
        
    for w_idx in range(4):
        i, j = divmod(w_idx, 2)
        C_idx = i * 4 + (j + 2)
        W[C_idx, 21:28] += W_s[w_idx, 0:7]
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We want to decompose A (2x4) * B (4x8) -> C (2x8)
    # Split B into B_L (4x4, cols 0..3) and B_R (4x4, cols 4..7)
    # C_L = A * B_L (2x4)
    # C_R = A * B_R (2x4)
    
    U_2x4, V_2x4, W_2x4 = _build_strassen_2x4()
    
    # U_2x4 shape: (8, 28)
    # V_2x4 shape: (16, 28)
    # W_2x4 shape: (8, 28)
    
    # For the full problem:
    # A is 2x4 (8 elements). U should be (8, rank).
    # B is 4x8 (32 elements). V should be (32, rank).
    # C is 2x8 (16 elements). W should be (16, rank).
    
    # Rank will be 28 + 28 = 56.
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Left half (cols 0..3 of B, cols 0..3 of C)
    # U for left half: A elements are the same.
    U[:, 0:28] = U_2x4
    
    # V for left half: B elements are rows 0..3, cols 0..3.
    # B index = i * p + j = i * 8 + j.
    # V_2x4 index = i * 4 + j.
    for i in range(4):
        for j in range(4):
            V[i * 8 + j, 0:28] = V_2x4[i * 4 + j, 0:28]
            
    # W for left half: C elements are rows 0..1, cols 0..3.
    # C index = k * n + i = k * 2 + i.
    # W_2x4 index = k * 4 + i? No, W_2x4 is for 2x4 output.
    # In _build_strassen_2x4, C was 2x4. Index c = k * 4 + i.
    # Here C is 2x8. Left half cols 0..3. Index c = k * 8 + i.
    for k in range(2):
        for i in range(4):
            W[i * 2 + k, 0:28] = W_2x4[i * 4 + k, 0:28]
            
    # Right half (cols 4..7 of B, cols 4..7 of C)
    # U for right half: A elements are the same.
    U[:, 28:56] = U_2x4
    
    # V for right half: B elements are rows 0..3, cols 4..7.
    for i in range(4):
        for j in range(4):
            V[i * 8 + (j + 4), 28:56] = V_2x4[i * 4 + j, 0:28]
            
    # W for right half: C elements are rows 0..1, cols 4..7.
    for k in range(2):
        for i in range(4):
            W[i * 2 + (k + 4), 28:56] = W_2x4[i * 4 + k, 0:28]
            
    # Convert to list of lists of floats/ints
    # The problem allows integer/half-integer entries.
    # Strassen uses +/-1, so these are integers.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
