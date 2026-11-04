import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2 x 2x2 matrix multiplication (rank 7).
    U, V, W have shape (4, 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Constructs rank-49 decomposition for 4x4 x 4x4.
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Define the 7 big Strassen multiplications
    # Each maps to a 2x2 sub-matrix operation
    big_mults = [
        # M1: (A11+A22)(B11+B22) -> C11, C22
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        # M2: (A21+A22)B11 -> C21, C22
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((1,0), 1), ((1,1), -1)]},
        # M3: A11(B12-B22) -> C12, C22
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((0,1), 1), ((1,1), 1)]},
        # M4: A22(B21-B11) -> C11, C12
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        # M5: (A11+A12)B22 -> C11, C12
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((0,1), 1)]},
        # M6: (A21-A11)(B11+B12) -> C22
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        # M7: (A12-A22)(B21+B22) -> C11
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]
    
    for big_idx, big in enumerate(big_mults):
        base_r = big_idx * 7
        for small_idx in range(7):
            r = base_r + small_idx
            
            # Fill U
            for (bi, bk), sign_A in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * bi + i2
                        k = 2 * bk + k2
                        U[i * 4 + k, r] += sign_A * U2[i2 * 2 + k2, small_idx]
                        
            # Fill V
            for (bk, bj), sign_B in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2 * bk + k2
                        j = 2 * bj + j2
                        V[k * 4 + j, r] += sign_B * V2[k2 * 2 + j2, small_idx]
                        
            # Fill W
            for (bj, bi), sign_C in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2 * bj + j2
                        i = 2 * bi + i2
                        W[j * 4 + i, r] += sign_C * W2[j2 * 2 + i2, small_idx]
                        
    return U, V, W

def _build_4x4x9_decomposition():
    """
    Builds decomposition for 4x4 x 4x9.
    Uses two 4x4x4 Strassen blocks (rank 98) and one 4x4x1 direct block (rank 16).
    Total rank 114.
    """
    U44, V44, W44 = _build_4x4x4_strassen()
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: Columns 0-3
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]
            
    # Block 2: Columns 4-7
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]
            
    # Block 3: Column 8 (Direct)
    for i in range(4):
        for k in range(4):
            r = 98 + i * 4 + k
            U[i * 4 + k, r] = 1
            V[k * 9 + 8, r] = 1
            W[8 * 4 + i, r] = 1
            
    return U, V, W, total_rank

def _validate(U, V, W, n, m, p, rank):
    """
    Validates the decomposition using a random test case.
    """
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Use a fixed seed for reproducibility in validation if needed, 
    # but random is fine for a quick check.
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute using tensor decomposition
    # a_vecs[i, r] = sum_k U[i, k, r] * A[i, k]
    a_vecs = np.einsum('ikr,ik->ir', U_reshaped, A_test)
    # b_vecs[j, r] = sum_k V[k, j, r] * B[k, j]
    b_vecs = np.einsum('kjr,kj->jr', V_reshaped, B_test)
    # C[i, j] = sum_r W[j, i, r] * a_vecs[i, r] * b_vecs[j, r]
    C_computed = np.einsum('jir,ir,jr->ij', W_reshaped, a_vecs, b_vecs)
    
    return np.allclose(C_computed, C_expected, atol=1e-9)

def _build_direct_fallback(n, m, p):
    """
    Direct decomposition with rank n*m*p.
    """
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
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Attempt the rank-114 decomposition
    try:
        U, V, W, total_rank = _build_4x4x9_decomposition()
        if _validate(U, V, W, n, m, p, total_rank):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to direct
    U, V, W, rank = _build_direct_fallback(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
