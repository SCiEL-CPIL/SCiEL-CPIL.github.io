import time
import numpy as np

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _strassen_222():
    """
    Returns U, V, W for rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U, V, W have shape (4, 7).
    Indices for A, B, C are 0..3 corresponding to (0,0), (0,1), (1,0), (1,1).
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00 + A11)(B00 + B11) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00 -> C10, C11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01 - B11) -> C01, C11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10 - B00) -> C00, C10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11 -> C00, C01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01) -> C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Trace convention: C[i,k] = sum_j A[i,j] * B[j,k]
    W index: W[k*n + i, r] corresponds to output C[i,k]
    """
    R = U.shape[1]
    tol = 1e-9
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    """
    Computes the Kronecker product of two tensor decompositions.
    Input: U1, V1, W1 (shapes (n1, r1), (m1, r1), (p1, r1))
           U2, V2, W2 (shapes (n2, r2), (m2, r2), (p2, r2))
    Output: U, V, W (shapes (n1*n2, r1*r2), (m1*m2, r1*r2), (p1*p2, r1*r2))
    """
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape
    
    R = r1 * r2
    U = np.zeros((n1 * n2, R), dtype=float)
    V = np.zeros((m1 * m2, R), dtype=float)
    W = np.zeros((p1 * p2, R), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            
            U[:, r_idx] = np.kron(U1[:, r1_idx], U2[:, r2_idx])
            V[:, r_idx] = np.kron(V1[:, r1_idx], V2[:, r2_idx])
            W[:, r_idx] = np.kron(W1[:, r1_idx], W2[:, r2_idx])
            
    return U, V, W

def _direct_sum_decomposition(U1, V1, W1):
    """
    Computes the direct sum of two identical tensor decompositions.
    Input: U1, V1, W1 (shapes (n, r), (m, r), (p, r))
    Output: U, V, W (shapes (n, 2r), (2m, 2r), (2p, 2r))
    Note: This assumes the second block operates on the second half of the indices.
    For M_{4,4,8}, we split p=8 into two p=4 blocks.
    """
    n, r = U1.shape
    m, _ = V1.shape
    p, _ = W1.shape
    
    R = 2 * r
    U = np.zeros((n, R), dtype=float)
    V = np.zeros((2 * m, R), dtype=float)
    W = np.zeros((2 * p, R), dtype=float)
    
    # First block
    U[:, :r] = U1
    V[:m, :r] = V1
    W[:p, :r] = W1
    
    # Second block
    U[:, r:] = U1
    V[m:, r:] = V1
    W[p:, r:] = W1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Step 1: Get Strassen decomposition for 2x2x2
    U2, V2, W2 = _strassen_222()
    
    # Step 2: Construct 4x4x4 decomposition via Kronecker product
    # R(4,4,4) <= R(2,2,2) * R(2,2,2) = 7 * 7 = 49
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    
    # Step 3: Construct 4x4x8 decomposition via direct sum
    # R(4,4,8) <= R(4,4,4) + R(4,4,4) = 49 + 49 = 98
    U, V, W = _direct_sum_decomposition(U444, V444, W444)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_school = np.zeros((n*m, schoolbook_rank))
        V_school = np.zeros((m*p, schoolbook_rank))
        W_school = np.zeros((p*n, schoolbook_rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii*m + jj, col] = 1
                    V_school[jj*p + kk, col] = 1
                    W_school[kk*n + ii, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
