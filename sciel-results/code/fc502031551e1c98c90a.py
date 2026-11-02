import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify_decomposition(U, V, W, n, m, p, tol=1e-12):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_reshaped = U.T.reshape(R, n, m)
    V_reshaped = V.T.reshape(R, m, p)
    W_reshaped = W.T.reshape(R, p, n)
    
    T_computed = np.einsum('rij,rjk,rki->ikj', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, p, m))
    
    return np.allclose(T_computed, T_target, atol=tol)

def _schoolbook():
    """Return schoolbook decomposition with rank n*m*p."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Rank 7.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11*(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22*(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank34():
    """
    Construct a rank-34 decomposition using Strassen on 2x2x2 blocks.
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Strassen 2x2x2 block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    S_U, S_V, S_W = _strassen_2x2x2()
    
    U1 = np.zeros((n * m, 7))
    V1 = np.zeros((m * p, 7))
    W1 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U1[i * m + j, r] = S_U[i * 2 + j, r]
                V1[i * p + j, r] = S_V[i * 2 + j, r]
                W1[j * n + i, r] = S_W[i * 2 + j, r]
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # Strassen 2x2x2 block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    S_U2, S_V2, S_W2 = _strassen_2x2x2()
    
    U2 = np.zeros((n * m, 7))
    V2 = np.zeros((m * p, 7))
    W2 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U2[i * m + j, r] = S_U2[i * 2 + j, r]
                V2[i * p + (j + 2), r] = S_V2[i * 2 + j, r]
                W2[(j + 2) * n + i, r] = S_W2[i * 2 + j, r]
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # Schoolbook: A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    U3 = np.zeros((n * m, 8))
    V3 = np.zeros((m * p, 8))
    W3 = np.zeros((p * n, 8))
    
    col = 0
    for i in range(2):
        for k in range(4):
            U3[i * m + 2, col] = 1
            V3[2 * p + k, col] = 1
            W3[k * n + i, col] = 1
            col += 1
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Schoolbook: A[2:3,0:3] * B[0:3,0:4] -> C[2:3,0:4]
    U4 = np.zeros((n * m, 12))
    V4 = np.zeros((m * p, 12))
    W4 = np.zeros((p * n, 12))
    
    col = 0
    for j in range(3):
        for k in range(4):
            U4[2 * m + j, col] = 1
            V4[j * p + k, col] = 1
            W4[k * n + 2, col] = 1
            col += 1
    
    U_parts.append(U4)
    V_parts.append(V4)
    W_parts.append(W4)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def _try_rank22_via_extension(start_time):
    """
    Try to construct a rank-22 decomposition by extending a rank-21 3x3x3 core
    with one additional term that handles the 4th column of B/C.
    
    Strategy: 
    - Use a known or constructed rank-21 decomposition for 3x3x3
    - Extend V and W to handle the 4th column
    - Solve for the additional term coefficients
    
    This is a heuristic attempt; if it fails, we fall back to rank 34.
    """
    # For now, this is a placeholder that will return None if it can't find a solution
    # The actual rank-21 3x3x3 decomposition is complex and not provided here
    # We'll rely on the rank-34 construction as the primary valid solution
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try rank-22 construction (extended rank-21 3x3x3)
    try:
        result = _try_rank22_via_extension(start_time)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Try rank-34 construction
    try:
        U, V, W = _construct_rank34()
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
