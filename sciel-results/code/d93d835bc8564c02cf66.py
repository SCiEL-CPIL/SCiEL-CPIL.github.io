import time
import numpy as np

DIMENSIONS = [2, 3, 3]
DEADLINE = 120.0  # Seconds, leaving margin for evaluator overhead

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _check_decomposition(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose tensor T."""
    a_dim, b_dim, c_dim = T.shape
    r = U.shape[1]
    # Reconstruct T using einsum for efficiency
    T_recon = np.einsum('as,bs,cs->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol, rtol=tol)

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _rank17_decomposition(n, m, p):
    """
    Construct a specific rank-17 decomposition for M_{2,3,3}.
    This uses a hybrid of Strassen's algorithm for the 2x2x2 sub-block
    and direct multiplication for the remaining terms.
    
    Indices:
    A is 2x3: A[i,j] -> U index i*3+j
       0: A00, 1: A01, 2: A02, 3: A10, 4: A11, 5: A12
    B is 3x3: B[j,k] -> V index j*3+k
       0: B00, 1: B01, 2: B02, 3: B10, 4: B11, 5: B12, 6: B20, 7: B21, 8: B22
    C is 2x3: C[i,k] -> W index k*2+i
       0: C00, 1: C10, 2: C01, 3: C11, 4: C02, 5: C12
    """
    rank = 17
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # --- Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] ---
    # A_sub = [[A00, A01], [A10, A11]] = [[U0, U1], [U3, U4]]
    # B_sub = [[B00, B01], [B10, B11]] = [[V0, V1], [V3, V4]]
    # C_sub = [[C00, C01], [C10, C11]] = [[W0, W2], [W1, W3]]
    #
    # Standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B11 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C10 = M2 + M4
    # C01 = M5 + M6
    # C11 = M1 - M2 + M3 - M6
    
    # Term 0: M1 = (U0+U4)(V0+V4)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = 1
    
    # Term 1: M2 = (U3+U4)V0
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = U0(V3-V0)
    U[0, 2] = 1
    V[3, 2] = 1; V[0, 2] = -1
    W[3, 2] = 1
    
    # Term 3: M4 = U4(V4-V0)
    U[4, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # Term 4: M5 = (U0+U1)V4
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # Term 5: M6 = (U3-U0)(V0+V1)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = -1
    
    # Term 6: M7 = (U1-U4)(V3+V4)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = -1
    
    # --- Part 2: Direct multiplication for column k=2 of B (B[0:2, 2]) ---
    # C[0,2] (W4) = A00*B02 + A01*B12
    # C[1,2] (W5) = A10*B02 + A11*B12
    # Term 7: A00*B02
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1
    
    # Term 8: A01*B12
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1
    
    # Term 9: A10*B02
    U[3, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # Term 10: A11*B12
    U[4, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    # --- Part 3: Direct multiplication for column j=2 of A (A[0:2, 2]) ---
    # C[0,0] (W0) += A02*B20
    # C[0,1] (W2) += A02*B21
    # C[0,2] (W4) += A02*B22
    # C[1,0] (W1) += A12*B20
    # C[1,1] (W3) += A12*B21
    # C[1,2] (W5) += A12*B22
    
    # Term 11: A02*B20
    U[2, 11] = 1
    V[6, 11] = 1
    W[0, 11] = 1
    
    # Term 12: A02*B21
    U[2, 12] = 1
    V[7, 12] = 1
    W[2, 12] = 1
    
    # Term 13: A02*B22
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # Term 14: A12*B20
    U[5, 14] = 1
    V[6, 14] = 1
    W[1, 14] = 1
    
    # Term 15: A12*B21
    U[5, 15] = 1
    V[7, 15] = 1
    W[3, 15] = 1
    
    # Term 16: A12*B22
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _try_rank14_via_kronecker(n, m, p):
    """
    Try to construct a rank-14 decomposition using Kronecker product structure.
    Group the 2x3 multiplication as 2 copies of 1x3 multiplication.
    Use rank-7 for 3x3 matrix multiplication if applicable, but here we're doing
    1x3 * 3x3 which is just vector-matrix multiplication (rank 9 schoolbook,
    but potentially lower).
    
    Actually, let's think differently. The 2x3 * 3x3 multiplication can be seen
    as computing C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2}.
    
    For each i, we compute a 1x3 vector times 3x3 matrix, which is rank 9 (schoolbook)
    or potentially lower. The total would be 2*9 = 18 if independent, but we can share
    structure.
    
    Let's try a different approach: use the fact that we can combine terms.
    """
    # This is a placeholder for potential rank-14 construction.
    # For now, return None to indicate not found.
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try rank 17 decomposition first
    U, V, W = _rank17_decomposition(n, m, p)
    T = _build_tensor(n, m, p)
    
    if _check_decomposition(U, V, W, T):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook if rank 17 fails (should not happen)
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
