import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # Deadline guard
    start_time = time.time()
    deadline = 120  # seconds

    # Strassen 2x2x2 coefficients
    # M0: (A00+A11)(B00+B11)
    # M1: (A10+A11)B00
    # M2: A00(B01-B11)
    # M3: A11(B10-B00)
    # M4: (A00+A01)B11
    # M5: (A10-A00)(B00+B01)
    # M6: (A01-A11)(B10+B11)
    
    sA_coefs = [
        [1, 0, 0, 1],  # M0
        [0, 0, 1, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 0, 1],  # M3
        [1, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5 (Fixed: A10 - A00)
        [0, 1, 0, -1]  # M6
    ]
    
    sB_coefs = [
        [1, 0, 0, 1],  # M0
        [1, 0, 0, 0],  # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, -1], # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, 1]   # M6
    ]
    
    sC_coefs = [
        [1, 0, 0, 0],   # M0 -> C00
        [0, 0, 1, -1],  # M1 -> C10, -C11
        [0, 1, 0, 1],   # M2 -> C01, C11
        [1, 0, 1, 0],   # M3 -> C00, C10
        [-1, 1, 0, 0],  # M4 -> -C00, C01
        [0, 0, 0, 1],   # M5 -> C11
        [1, 0, 0, 0]    # M6 -> C00
    ]
    
    # Part 1: Strassen for 4x4x4 (columns 0-3 of B)
    # Rank 49
    U1 = np.zeros((16, 49), dtype=np.int8)
    V1 = np.zeros((28, 49), dtype=np.int8)
    W1 = np.zeros((28, 49), dtype=np.int8)
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            # A indices
            for i in range(4):
                for j in range(4):
                    block_a = (i//2)*2 + (j//2)
                    local_a = (i%2)*2 + (j%2)
                    coef = sA_coefs[t][block_a] * sA_coefs[s][local_a]
                    if coef != 0:
                        U1[i*4+j, r] = coef
            
            # B indices (k=0..3)
            for j in range(4):
                for k in range(4):
                    block_b = (j//2)*2 + (k//2)
                    local_b = (j%2)*2 + (k%2)
                    coef = sB_coefs[t][block_b] * sB_coefs[s][local_b]
                    if coef != 0:
                        V1[j*7+k, r] = coef
            
            # C indices (k=0..3)
            for i in range(4):
                for k in range(4):
                    block_c = (k//2)*2 + (i//2)
                    local_c = (k%2)*2 + (i%2)
                    coef = sC_coefs[t][block_c] * sC_coefs[s][local_c]
                    if coef != 0:
                        W1[k*4+i, r] = coef
                        
    # Part 2: Strassen-packed 4x4x3 (columns 4-6 of B)
    # We decompose the 4x4x3 multiplication into 4 independent 2x2x3 multiplications.
    # The 2x2x3 multiplication can be done with rank 11 (Strassen 2x2x2 + Schoolbook 2x2x1).
    # Total rank for Part 2 = 4 * 11 = 44.
    
    # Helper to generate 2x2x2 Strassen coefficients for a specific block
    def get_2x2x2_strassen(block_idx_a, block_idx_b, block_idx_c):
        # This is a bit tricky because the indices are local to the 2x2 block.
        # We will just use the standard Strassen coefficients and map them.
        # But wait, the "block" here refers to the 2x2 sub-blocks of the larger matrices.
        # The logic is the same as in Part 1, just for a 2x2x2 tensor.
        pass

    # Let's manually construct the 2x2x3 decomposition for one block and then replicate.
    # The 4 blocks are:
    # 1. A[0:2, 0:2] * B[0:2, 4:7] -> C[0:2, 4:7]
    # 2. A[0:2, 2:4] * B[2:4, 4:7] -> C[0:2, 4:7]
    # 3. A[2:4, 0:2] * B[0:2, 4:7] -> C[2:4, 4:7]
    # 4. A[2:4, 2:4] * B[2:4, 4:7] -> C[2:4, 4:7]
    
    # For each block, we need a rank-11 decomposition of 2x2x3.
    # A 2x2x3 multiplication is (A 2x2) * (B 2x3) = C 2x3.
    # We can split B into B_left (2x2) and B_right (2x1).
    # A * B_left is 2x2x2 (Strassen, rank 7).
    # A * B_right is 2x2x1 (Schoolbook, rank 4).
    # Total rank 11.
    
    # Let's generate the 2x2x3 decomposition coefficients.
    # Local indices: i,j in 0..1 for A, k in 0..2 for B/C columns.
    # B columns 0,1 form the 2x2 part, column 2 is the 2x1 part.
    
    # Strassen 2x2x2 part (for B cols 0,1)
    sA_22 = sA_coefs
    sB_22 = sB_coefs
    sC_22 = sC_coefs
    
    # Schoolbook 2x2x1 part (for B col 2)
    # C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2]
    # Rank 4:
    # r=0: A[0,0]B[0,2] -> C[0,2]
    # r=1: A[0,1]B[1,2] -> C[0,2]
    # r=2: A[1,0]B[0,2] -> C[1,2]
    # r=3: A[1,1]B[1,2] -> C[1,2]
    
    U_223 = np.zeros((4, 11), dtype=np.int8)
    V_223 = np.zeros((6, 11), dtype=np.int8)
    W_223 = np.zeros((6, 11), dtype=np.int8)
    
    # First 7 terms: Strassen 2x2x2
    for t in range(7):
        r = t
        for i in range(2):
            for j in range(2):
                coef = sA_22[t][i*2+j]
                if coef != 0:
                    U_223[i*2+j, r] = coef
        for j in range(2):
            for k in range(2): # k=0,1 for B cols
                coef = sB_22[t][j*2+k]
                if coef != 0:
                    V_223[j*3+k, r] = coef
        for i in range(2):
            for k in range(2): # k=0,1 for C cols
                coef = sC_22[t][k*2+i]
                if coef != 0:
                    W_223[k*2+i, r] = coef
                    
    # Next 4 terms: Schoolbook 2x2x1
    for t in range(4):
        r = 7 + t
        i, j = divmod(t, 2) # t=0:(0,0), t=1:(0,1), t=2:(1,0), t=3:(1,1)
        U_223[i*2+j, r] = 1
        V_223[j*3+2, r] = 1 # B col 2
        W_223[2*2+i, r] = 1 # C col 2, row i
        
    # Now replicate this 2x2x3 decomposition for the 4 blocks.
    U2 = np.zeros((16, 44), dtype=np.int8)
    V2 = np.zeros((28, 44), dtype=np.int8)
    W2 = np.zeros((28, 44), dtype=np.int8)
    
    block_defs = [
        (0, 0, 0), # A[0:2, 0:2], B[0:2, 4:7], C[0:2, 4:7]
        (0, 2, 0), # A[0:2, 2:4], B[2:4, 4:7], C[0:2, 4:7]
        (2, 0, 2), # A[2:4, 0:2], B[0:2, 4:7], C[2:4, 4:7]
        (2, 2, 2)  # A[2:4, 2:4], B[2:4, 4:7], C[2:4, 4:7]
    ]
    
    for b_idx, (a_row, a_col, c_row) in enumerate(block_defs):
        offset = b_idx * 11
        for r in range(11):
            global_r = offset + r
            for i in range(2):
                for j in range(2):
                    coef = U_223[i*2+j, r]
                    if coef != 0:
                        U2[(a_row+i)*4 + (a_col+j), global_r] = coef
            for j in range(2):
                for k in range(3):
                    coef = V_223[j*3+k, r]
                    if coef != 0:
                        V2[(a_col+j)*7 + (4+k), global_r] = coef
            for i in range(2):
                for k in range(3):
                    coef = W_223[k*2+i, r]
                    if coef != 0:
                        W2[(4+k)*4 + (c_row+i), global_r] = coef
                        
    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Verify
    T = np.zeros((16, 28, 28), dtype=np.int8)
    for i in range(4):
        for j in range(4):
            for k in range(7):
                T[i*4+j, j*7+k, k*4+i] = 1
                
    T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    if not np.array_equal(T_approx, T):
        # Fallback to schoolbook if verification fails
        U = np.zeros((16, 112), dtype=np.int8)
        V = np.zeros((28, 112), dtype=np.int8)
        W = np.zeros((28, 112), dtype=np.int8)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, col] = 1
                    V[j*7+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
