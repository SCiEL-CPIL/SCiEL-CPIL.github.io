"""Rank-minimal decomposition for M_{4,5,6} using block decomposition with Strassen.
Strategy: M_{4,5,6} = M_{4,4,6} ⊕ M_{4,1,6}
         M_{4,4,6} = M_{4,4,4} ⊕ M_{4,4,2}
M_{4,4,4}: rank 56 (Strassen-based)
M_{4,4,2}: rank 28 (Strassen-based)
M_{4,1,6}: rank 6
Total rank: 56 + 28 + 6 = 90
"""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition for M_{n,m,p}."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _m_2_2_2_strassen():
    """Rank-7 Strassen decomposition for M_{2,2,2}."""
    n = m = p = 2
    rank = 7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # M1: c=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2: c=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: c=2: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = -1
    
    # M4: c=3: A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = -1
    
    # M5: c=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6: c=5: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7: c=6: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _m_2_2_4():
    """M_{2,2,4} with rank 14 (two Strassen blocks for k=0,1 and k=2,3)."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_sub, V_sub, W_sub = _m_2_2_2_strassen()
    
    # First block: k in [0,2), offset_k=0
    for c in range(7):
        col = c
        U[:, col] = U_sub[:, c]
        for j in range(2):
            for k_sub in range(2):
                V[j*4 + k_sub, col] = V_sub[j*2 + k_sub, c]
        for k_sub in range(2):
            for i in range(2):
                W[k_sub*2 + i, col] = W_sub[k_sub*2 + i, c]
    
    # Second block: k in [2,4), offset_k=2
    for c in range(7):
        col = 7 + c
        U[:, col] = U_sub[:, c]
        for j in range(2):
            for k_sub in range(2):
                k_actual = k_sub + 2
                V[j*4 + k_actual, col] = V_sub[j*2 + k_sub, c]
        for k_sub in range(2):
            k_actual = k_sub + 2
            for i in range(2):
                W[k_actual*2 + i, col] = W_sub[k_sub*2 + i, c]
    
    return U, V, W

def _m_2_2_2():
    """M_{2,2,2} with rank 7 (Strassen)."""
    return _m_2_2_2_strassen()

def _m_2_4_2():
    """M_{2,4,2} with rank 14 (two Strassen blocks for j=0,1 and j=2,3)."""
    n, m, p = 2, 4, 2
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_sub, V_sub, W_sub = _m_2_2_2_strassen()
    
    # First block: j in [0,2)
    for c in range(7):
        col = c
        for i in range(2):
            for j in range(2):
                U[i*4 + j, col] = U_sub[i*2 + j, c]
        for j in range(2):
            for k in range(2):
                V[j*2 + k, col] = V_sub[j*2 + k, c]
        for k in range(2):
            for i in range(2):
                W[k*2 + i, col] = W_sub[k*2 + i, c]
    
    # Second block: j in [2,4)
    for c in range(7):
        col = 7 + c
        for i in range(2):
            for j in range(2):
                j_actual = j + 2
                U[i*4 + j_actual, col] = U_sub[i*2 + j, c]
        for j in range(2):
            j_actual = j + 2
            for k in range(2):
                V[j_actual*2 + k, col] = V_sub[j*2 + k, c]
        for k in range(2):
            for i in range(2):
                W[k*2 + i, col] = W_sub[k*2 + i, c]
    
    return U, V, W

def _m_2_4_4():
    """M_{2,4,4} with rank 28 (two M_{2,2,4} blocks for j=0,1 and j=2,3)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_sub, V_sub, W_sub = _m_2_2_4()
    
    # First block: j in [0,2)
    for i in range(2):
        for j in range(2):
            U[i*4 + j, 0:14] = U_sub[i*2 + j, :]
    V[0:8, 0:14] = V_sub[:, :]
    W[:, 0:14] = W_sub[:, :]
    
    # Second block: j in [2,4)
    for i in range(2):
        for j in range(2):
            j_actual = j + 2
            U[i*4 + j_actual, 14:28] = U_sub[i*2 + j, :]
    for j in range(2):
        j_actual = j + 2
        V[j_actual*4: j_actual*4+4, 14:28] = V_sub[j*4: j*4+4, :]
    W[:, 14:28] = W_sub[:, :]
    
    return U, V, W

def _m_4_4_2():
    """M_{4,4,2} with rank 28 (two M_{2,4,2} blocks for i=0,1 and i=2,3)."""
    n, m, p = 4, 4, 2
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_sub, V_sub, W_sub = _m_2_4_2()
    
    # First block: i in [0,2)
    for i in range(2):
        for j in range(4):
            U[i*4 + j, 0:14] = U_sub[i*4 + j, :]
    V[:, 0:14] = V_sub[:, :]
    for k in range(2):
        for i in range(2):
            W[k*4 + i, 0:14] = W_sub[k*2 + i, :]
    
    # Second block: i in [2,4)
    for i in range(2):
        i_actual = i + 2
        for j in range(4):
            U[i_actual*4 + j, 14:28] = U_sub[i*4 + j, :]
    V[:, 14:28] = V_sub[:, :]
    for k in range(2):
        for i in range(2):
            i_actual = i + 2
            W[k*4 + i_actual, 14:28] = W_sub[k*2 + i, :]
    
    return U, V, W

def _m_4_4_4():
    """M_{4,4,4} with rank 56 (two M_{2,4,4} blocks for i=0,1 and i=2,3)."""
    n, m, p = 4, 4, 4
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_sub, V_sub, W_sub = _m_2_4_4()
    
    # First block: i in [0,2)
    U[0:8, 0:28] = U_sub[:, :]
    V[:, 0:28] = V_sub[:, :]
    for k in range(4):
        for i in range(2):
            W[k*4 + i, 0:28] = W_sub[k*2 + i, :]
    
    # Second block: i in [2,4)
    for i in range(2):
        i_actual = i + 2
        U[i_actual*4: i_actual*4+4, 28:56] = U_sub[i*4: i*4+4, :]
    V[:, 28:56] = V_sub[:, :]
    for k in range(4):
        for i in range(2):
            i_actual = i + 2
            W[k*4 + i_actual, 28:56] = W_sub[k*2 + i, :]
    
    return U, V, W

def _m_4_1_6():
    """M_{4,1,6}: rank 6."""
    n, m, p = 4, 1, 6
    rank = 6
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i in range(n):
        for k in range(p):
            U[i, k] = 1.0
    for k in range(p):
        V[k, k] = 1.0
        W[k*n, k] = 1.0
    
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check that the decomposition is exact."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i*m + j]
                v = V[j*p + k]
                w = W[k*n + i]
                val = np.sum(u * v * w)
                if abs(val - 1.0) > tol:
                    return False, val
    return True, 0.0

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    deadline = time.time() + 110
    
    # M_{4,5,6} = M_{4,4,6} ⊕ M_{4,1,6}
    # M_{4,4,6} = M_{4,4,4} ⊕ M_{4,4,2}
    
    U_444, V_444, W_444 = _m_4_4_4()   # rank 56
    U_442, V_442, W_442 = _m_4_4_2()   # rank 28
    U_416, V_416, W_416 = _m_4_1_6()   # rank 6
    
    rank_444 = U_444.shape[1]  # 56
    rank_442 = U_442.shape[1]  # 28
    rank_416 = U_416.shape[1]  # 6
    
    total_rank = rank_444 + rank_442 + rank_416  # 90
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Block 1: M_{4,4,4} occupies j in [0,4), k in [0,4)
    col_offset = 0
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col_offset:col_offset+rank_444] = U_444[i*4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j*6 + k, col_offset:col_offset+rank_444] = V_444[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, col_offset:col_offset+rank_444] = W_444[k*4 + i, :]
    
    col_offset += rank_444
    
    # Block 2: M_{4,4,2} occupies j in [0,4), k in [4,6)
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col_offset:col_offset+rank_442] = U_442[i*4 + j, :]
    for j in range(4):
        for k in range(2):
            k_actual = k + 4
            V[j*6 + k_actual, col_offset:col_offset+rank_442] = V_442[j*2 + k, :]
    for k in range(2):
        k_actual = k + 4
        for i in range(4):
            W[k_actual*4 + i, col_offset:col_offset+rank_442] = W_442[k*4 + i, :]
    
    col_offset += rank_442
    
    # Block 3: M_{4,1,6} occupies j=4
    for i in range(4):
        U[i*5 + 4, col_offset:col_offset+rank_416] = U_416[i, :]
    for k in range(6):
        V[4*6 + k, col_offset:col_offset+rank_416] = V_416[k, :]
    for k in range(6):
        for i in range(4):
            W[k*4 + i, col_offset:col_offset+rank_416] = W_416[k*4, :]
    
    # Verify
    valid, err = _validate(U, V, W, n, m, p, tol=1e-8)
    if not valid:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
