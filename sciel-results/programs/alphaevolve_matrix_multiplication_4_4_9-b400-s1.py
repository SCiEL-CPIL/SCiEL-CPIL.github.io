import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    # Strassen: 7 multiplications
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    
    # U: shape (4,7), U[i*2+k, r]
    # V: shape (4,7), V[k*2+j, r]
    # W: shape (4,7), W[j*2+i, r]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A part: A11=(i=0,k=0), A22=(i=1,k=1)
    # B part: B11=(k=0,j=0), B22=(k=1,j=1)
    # W part: C11 gets +M1, C22 gets +M1
    # C11: i=0,j=0 -> W[0*2+0, 0] = W[0,0]
    # C22: i=1,j=1 -> W[1*2+1, 0] = W[3,0]
    U[0, 0] = 1  # A11: i=0,k=0
    U[3, 0] = 1  # A22: i=1,k=1
    V[0, 0] = 1  # B11: k=0,j=0
    V[3, 0] = 1  # B22: k=1,j=1
    W[0, 0] = 1  # C11: j=0,i=0
    W[3, 0] = 1  # C22: j=1,i=1
    
    # M2 = (A21+A22)B11
    # A part: A21=(i=1,k=0), A22=(i=1,k=1)
    # B part: B11=(k=0,j=0)
    # C part: C21 gets +M2, C22 gets -M2
    # C21: i=1,j=0 -> W[0*2+1, 1] = W[1,1]
    # C22: i=1,j=1 -> W[1*2+1, 1] = W[3,1]
    U[2, 1] = 1  # A21: i=1,k=0
    U[3, 1] = 1  # A22: i=1,k=1
    V[0, 1] = 1  # B11: k=0,j=0
    W[1, 1] = 1  # C21: j=0,i=1
    W[3, 1] = -1 # C22: j=1,i=1
    
    # M3 = A11(B12-B22)
    # A part: A11=(i=0,k=0)
    # B part: B12=(k=0,j=1), B22=(k=1,j=1)
    # C part: C12 gets +M3, C22 gets +M3
    # C12: i=0,j=1 -> W[1*2+0, 2] = W[2,2]
    # C22: i=1,j=1 -> W[1*2+1, 2] = W[3,2]
    U[0, 2] = 1  # A11: i=0,k=0
    V[1, 2] = 1  # B12: k=0,j=1
    V[3, 2] = -1 # B22: k=1,j=1
    W[2, 2] = 1  # C12: j=1,i=0
    W[3, 2] = 1  # C22: j=1,i=1
    
    # M4 = A22(B21-B11)
    # A part: A22=(i=1,k=1)
    # B part: B21=(k=1,j=0), B11=(k=0,j=0)
    # C part: C11 gets +M4, C21 gets +M4
    # C11: i=0,j=0 -> W[0*2+0, 3] = W[0,3]
    # C21: i=1,j=0 -> W[0*2+1, 3] = W[1,3]
    U[3, 3] = 1  # A22: i=1,k=1
    V[2, 3] = 1  # B21: k=1,j=0
    V[0, 3] = -1 # B11: k=0,j=0
    W[0, 3] = 1  # C11: j=0,i=0
    W[1, 3] = 1  # C21: j=0,i=1
    
    # M5 = (A11+A12)B22
    # A part: A11=(i=0,k=0), A12=(i=0,k=1)
    # B part: B22=(k=1,j=1)
    # C part: C11 gets -M5, C12 gets +M5
    # C11: i=0,j=0 -> W[0*2+0, 4] = W[0,4]
    # C12: i=0,j=1 -> W[1*2+0, 4] = W[2,4]
    U[0, 4] = 1  # A11: i=0,k=0
    U[1, 4] = 1  # A12: i=0,k=1
    V[3, 4] = 1  # B22: k=1,j=1
    W[0, 4] = -1 # C11: j=0,i=0
    W[2, 4] = 1  # C12: j=1,i=0
    
    # M6 = (A21-A11)(B11+B12)
    # A part: A21=(i=1,k=0), A11=(i=0,k=0)
    # B part: B11=(k=0,j=0), B12=(k=0,j=1)
    # C part: C22 gets +M6
    # C22: i=1,j=1 -> W[1*2+1, 5] = W[3,5]
    U[2, 5] = 1  # A21: i=1,k=0
    U[0, 5] = -1 # A11: i=0,k=0
    V[0, 5] = 1  # B11: k=0,j=0
    V[1, 5] = 1  # B12: k=0,j=1
    W[3, 5] = 1  # C22: j=1,i=1
    
    # M7 = (A12-A22)(B21+B22)
    # A part: A12=(i=0,k=1), A22=(i=1,k=1)
    # B part: B21=(k=1,j=0), B22=(k=1,j=1)
    # C part: C11 gets +M7
    # C11: i=0,j=0 -> W[0*2+0, 6] = W[0,6]
    U[1, 6] = 1  # A12: i=0,k=1
    U[3, 6] = -1 # A22: i=1,k=1
    V[2, 6] = 1  # B21: k=1,j=0
    V[3, 6] = 1  # B22: k=1,j=1
    W[0, 6] = 1  # C11: j=0,i=0
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    Uses Strassen at the 4x4 level (splitting into 2x2 blocks),
    with each 2x2 x 2x2 multiplication done via Strassen (rank 7).
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    # At the 4x4 level, we have 7 "big multiplications"
    # Each big multiplication is a 2x2 x 2x2 multiplication
    # We need to map the 2x2 block structure to 4x4 indices
    
    # 4x4 A split into 2x2 blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # For the 4x4 Strassen, the "A matrix" in each 2x2 multiplication
    # is a 2x2 matrix of 2x2 blocks. So A11_block, A12_block, etc.
    
    # The 7 Strassen combinations at the 4x4 level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each of these is a 2x2 x 2x2 multiplication with rank 7
    # So total rank = 7 * 7 = 49
    
    # For each of the 7 big multiplications, we need to:
    # 1. Determine which 2x2 blocks of A are involved (with signs)
    # 2. Determine which 2x2 blocks of B are involved (with signs)
    # 3. Determine which 2x2 blocks of C get the result (with signs)
    
    # Let me define the block structure:
    # Block (bi, bk) of A: rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
    # Block (bk, bj) of B: rows 2*bk to 2*bk+1, cols 2*bj to 2*bj+1
    # Block (bj, bi) of C: rows 2*bj to 2*bj+1, cols 2*bi to 2*bi+1
    
    # Strassen at 4x4 level (blocks are 2x2):
    # bigM1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # bigM2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    # bigM3 = A11(B12-B22) -> C12 += M3, C22 += M3
    # bigM4 = A22(B21-B11) -> C11 += M4, C21 += M4
    # bigM5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    # bigM6 = (A21-A11)(B11+B12) -> C22 += M6
    # bigM7 = (A12-A22)(B21+B22) -> C11 += M7
    
    # For each big multiplication bigM_r, we use the 2x2 Strassen (U2, V2, W2)
    # The 2x2 indices within each block map to 4x4 indices as:
    # A block (bi, bk): 4x4 index (i,k) where i = 2*bi + i2, k = 2*bk + k2
    # B block (bk, bj): 4x4 index (k,j) where k = 2*bk + k2, j = 2*bj + j2
    # C block (bj, bi): 4x4 index (j,i) where j = 2*bj + j2, i = 2*bi + i2
    
    U = np.zeros((16, 49))
    V = np.zeros((36, 49))  # Wait, for 4x4 x 4x4, V should be (4*4, 49) = (16, 49)
    # Actually for the 4x4 x 4x4 sub-problem, V is (m*p, rank) = (4*4, 49) = (16, 49)
    W = np.zeros((16, 49))
    
    # Define the 7 big Strassen multiplications
    # Each entry: (A_blocks, B_blocks, C_blocks)
    # A_blocks: list of (block_row, block_col, sign)
    # B_blocks: list of (block_row, block_col, sign)
    # C_blocks: list of (block_row, block_col, sign)
    
    big_mults = [
        # M1 = (A11+A22)(B11+B22)
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # M2 = (A21+A22)B11
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((0,1), 1), ((1,1), -1)]
        },
        # M3 = A11(B12-B22)
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((1,0), 1), ((1,1), 1)]
        },
        # M4 = A22(B21-B11)
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((0,1), 1)]
        },
        # M5 = (A11+A12)B22
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((1,0), 1)]
        },
        # M6 = (A21-A11)(B11+B12)
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        # M7 = (A12-A22)(B21+B22)
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        },
    ]
    
    for big_idx, big in enumerate(big_mults):
        base_r = big_idx * 7
        
        for small_idx in range(7):
            r = base_r + small_idx
            
            # U: for each A block involved, add the U2 contribution
            for (bi, bk), sign_A in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * bi + i2
                        k = 2 * bk + k2
                        U[i * 4 + k, r] += sign_A * U2[i2 * 2 + k2, small_idx]
            
            # V: for each B block involved, add the V2 contribution
            for (bk, bj), sign_B in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2 * bk + k2
                        j = 2 * bj + j2
                        V[k * 4 + j, r] += sign_B * V2[k2 * 2 + j2, small_idx]
            
            # W: for each C block involved, add the W2 contribution
            for (bj, bi), sign_C in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2 * bj + j2
                        i = 2 * bi + i2
                        W[j * 4 + i, r] += sign_C * W2[j2 * 2 + i2, small_idx]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Build 4x4 x 4x4 Strassen decomposition (rank 49)
    U44, V44, W44 = _build_4x4x4_strassen()
    
    # Build 4x4 x 4x1 schoolbook decomposition (rank 16)
    # C[i,0] = sum_k A[i,k] * B[k,0]
    U41 = np.zeros((16, 16))
    V41 = np.zeros((4, 16))  # m*p = 4*1 = 4
    W41 = np.zeros((4, 16))  # p*n = 1*4 = 4
    
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U41[i * 4 + k, r] = 1
            V41[k * 1 + 0, r] = 1
            W41[0 * 4 + i, r] = 1
    
    # Now combine for 4x4 x 4x9:
    # Split B into B1 (4x4) and B2 (4x5)
    # C = A*B1 + A*B2
    # A*B1: rank 49 (Strassen)
    # A*B2: 4x4 x 4x5 = 4x4 x 4x4 + 4x4 x 4x1, rank 49 + 16 = 65
    # Total rank: 49 + 65 = 114
    
    # Actually, let me split as B1 (4x4) + B2 (4x4) + B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # Rank: 49 + 49 + 16 = 114
    
    total_rank = 49 + 49 + 16
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))  # m*p = 4*9 = 36
    W = np.zeros((36, total_rank))  # p*n = 9*4 = 36
    
    # Part 1: A*B1 where B1 is columns 0-3 of B
    # U44, V44, W44 are for 4x4 x 4x4
    # V44 is indexed by (k, j) where j is 0-3 (local)
    # In the full problem, j is 0-3 (global)
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
    
    # Part 2: A*B2 where B2 is columns 4-7 of B
    # Same U44, but V and W shifted
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
    
    # Part 3: A*B3 where B3 is column 8 of B
    # U41, V41, W41 are for 4x4 x 4x1
    U[:, 98:114] = U41
    for k in range(4):
        V[k * 9 + 8, 98:114] = V41[k, :]
    for i in range(4):
        W[8 * 4 + i, 98:114] = W41[i, :]
    
    # Validate
    # Check that the tensor decomposition is correct
    # T[i,k,j] should equal sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # For matrix multiplication, T[i,k,j] = 1 for all valid (i,k,j)
    
    # Let's verify with a random test
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 9))
    for r in range(total_rank):
        # Extract A_part and B_part from U and V
        A_part = np.zeros((4, 4))
        B_part = np.zeros((4, 9))
        for i in range(4):
            for k in range(4):
                A_part[i, k] = U[i * 4 + k, r]
        for k in range(4):
            for j in range(9):
                B_part[k, j] = V[k * 9 + j, r]
        
        C_contrib = A_part @ B_part
        for i in range(4):
            for j in range(9):
                C_computed[i, j] += W[j * 4 + i, r] * C_contrib[i, j]
    
    # This verification is O(rank * n * m * p) which is 114 * 4 * 4 * 9 = 16416, fast enough
    # But the above is not quite right. Let me think...
    
    # Actually, the decomposition means:
    # C[i,j] = sum_k A[i,k] * B[k,j] = sum_r (sum_k U[i*m+k,r] * A[i,k]) * (sum_k V[k*p+j,r] * B[k,j]) * W[j*n+i,r]
    
    # No wait, that's not right either. The tensor decomposition is:
    # T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # where T[i,k,j] = 1 (the multiplication tensor)
    
    # So C[i,j] = sum_k A[i,k] * B[k,j] = sum_k sum_r T[i,k,j] * A[i,k] * B[k,j]
    # = sum_r sum_k U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r] * A[i,k] * B[k,j]
    # = sum_r W[j*n+i,r] * (sum_k U[i*m+k,r] * A[i,k]) * (sum_k V[k*p+j,r] * B[k,j])
    
    # So let me verify this way:
    C_computed2 = np.zeros((4, 9))
    for r in range(total_rank):
        a_vec = np.zeros(4)
        b_vec = np.zeros(9)
        for i in range(4):
            a_vec[i] = sum(U[i * 4 + k, r] * A_test[i, k] for k in range(4))
        for j in range(9):
            b_vec[j] = sum(V[k * 9 + j, r] * B_test[k, j] for k in range(4))
        for i in range(4):
            for j in range(9):
                C_computed2[i, j] += W[j * 4 + i, r] * a_vec[i] * b_vec[j]
    
    if not np.allclose(C_computed2, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    U[i * m + k, idx] = 1
                    V[k * p + j, idx] = 1
                    W[j * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
