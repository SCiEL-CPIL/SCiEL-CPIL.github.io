"""Matrix multiplication 3x4x5 tensor decomposition - rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, each shape (4, 7).
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    rank = 7
    U = np.zeros((4, rank), dtype=float)
    V = np.zeros((4, rank), dtype=float)
    W = np.zeros((4, rank), dtype=float)
    
    # U[i*2+j, c] = coefficient of A_{i,j} in factor c
    # V[j*2+k, c] = coefficient of B_{j,k} in factor c
    # W[k*2+i, c] = coefficient of C_{k,i} in factor c (trace convention)
    
    # Factor 0: M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Factor 1: M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Factor 2: M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # Factor 3: M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Factor 4: M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Factor 5: M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = -1
    
    # Factor 6: M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if j == k else 0.0
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - expected) > tol:
                    return False
    return True


def _build_rank51():
    """Build rank-51 decomposition using three disjoint Strassen 2x2x2 blocks."""
    n, m, p = 3, 4, 5
    total_rank = 51
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    # Block 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    U1, V1, W1 = _strassen_2x2x2()
    for i in range(2):
        for j in range(2):
            U[i * m + j, col:col+7] = U1[i * 2 + j, :]
    for j in range(2):
        for k in range(2):
            V[j * p + k, col:col+7] = V1[j * 2 + k, :]
    for k in range(2):
        for i in range(2):
            W[k * n + i, col:col+7] = W1[k * 2 + i, :]
    col += 7
    
    # Block 2: Strassen for i in {0,1}, j in {2,3}, k in {2,3}
    U2, V2, W2 = _strassen_2x2x2()
    for i in range(2):
        for j in range(2):
            U[i * m + (j + 2), col:col+7] = U2[i * 2 + j, :]
    for j in range(2):
        for k in range(2):
            V[(j + 2) * p + (k + 2), col:col+7] = V2[j * 2 + k, :]
    for k in range(2):
        for i in range(2):
            W[(k + 2) * n + i, col:col+7] = W2[k * 2 + i, :]
    col += 7
    
    # Block 3: Strassen for i in {0,1}, j in {2,3}, k in {0,1}
    U3, V3, W3 = _strassen_2x2x2()
    for i in range(2):
        for j in range(2):
            U[i * m + (j + 2), col:col+7] = U3[i * 2 + j, :]
    for j in range(2):
        for k in range(2):
            V[(j + 2) * p + k, col:col+7] = V3[j * 2 + k, :]
    for k in range(2):
        for i in range(2):
            W[k * n + i, col:col+7] = W3[k * 2 + i, :]
    col += 7
    
    # Schoolbook for remaining entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in Strassen block 1
                if i < 2 and j < 2 and k < 2:
                    continue
                # Skip if in Strassen block 2
                if i < 2 and 2 <= j < 4 and 2 <= k < 4:
                    continue
                # Skip if in Strassen block 3
                if i < 2 and 2 <= j < 4 and k < 2:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank-51 Strassen-based decomposition (3 disjoint blocks)
    try:
        U, V, W = _build_rank51()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    if time.time() > deadline:
        U, V, W = _schoolbook_decomposition(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
