DIMENSIONS = [2, 2, 4]
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x4():
    """
    Deterministic Strassen-based rank-14 decomposition for 2x2 times 2x4.
    
    The 2x4 multiplication can be viewed as two independent 2x2 multiplications:
    - First 2x2: columns 0,1 of the 2x4 matrix
    - Second 2x2: columns 2,3 of the 2x4 matrix
    
    Each 2x2 multiplication uses the Strassen algorithm with rank 7.
    Total rank = 7 + 7 = 14.
    
    Strassen's 7 multiplications for 2x2:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    Result:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factor matrices (4x7 each)
    # U rows: (i*m+j) for i in {0,1}, j in {0,1} -> indices 0,1,2,3
    U_2x2 = np.array([
        # i=0, j=0 (A11): M1, M3, M5, M6, M7
        [1, 1, 0, 0, 0, 1, 1],
        # i=0, j=1 (A12): M5, M7
        [0, 0, 1, 0, 1, 0, 0],
        # i=1, j=0 (A21): M2, M6
        [0, 1, 0, 0, 0, -1, 0],
        # i=1, j=1 (A22): M1, M2, M4, M7
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    # V rows: (j*p+k) for j in {0,1}, k in {0,1} -> indices 0,1,2,3
    V_2x2 = np.array([
        # j=0, k=0 (B11): M1, M2, M4, M6
        [1, 1, 0, -1, 0, 1, 0],
        # j=0, k=1 (B12): M5, M6
        [1, 0, 0, 1, 1, 1, 0],
        # j=1, k=0 (B21): M1, M3, M7
        [1, 1, -1, 0, 0, 0, 1],
        # j=1, k=1 (B22): M1, M3, M5, M7
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    # W rows: (k*n+i) for k in {0,1}, i in {0,1} -> indices 0,1,2,3
    # C00 = M1 + M4 - M5 + M7  (k=0, i=0)
    # C01 = M3 + M5             (k=1, i=0)
    # C10 = M2 + M4             (k=0, i=1)
    # C11 = M1 - M2 + M3 + M6   (k=1, i=1)
    W_2x2 = np.array([
        # k=0, i=0 (C00): M1 + M4 - M5 + M7
        [1, 0, 0, 1, -1, 0, 1],
        # k=1, i=0 (C01): M3 + M5
        [0, 0, 1, 0, 1, 0, 0],
        # k=0, i=1 (C10): M2 + M4
        [0, 1, 0, 1, 0, 0, 0],
        # k=1, i=1 (C11): M1 - M2 + M3 + M6
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1: first 2 columns (k=0,1)
    U[:, 0:7] = U_2x2
    # For the first block, V rows correspond to j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
    # In the full 2x4, j is the row index of the 2x4 matrix, k is the column index
    # Block 1 uses columns 0,1 of the 2x4 matrix
    V[0*4+0:0*4+2, 0:7] = V_2x2[0:2, :]  # j=0, k=0,1
    V[1*4+0:1*4+2, 0:7] = V_2x2[2:4, :]  # j=1, k=0,1
    W[0*2+0:0*2+2, 0:7] = W_2x2[0:2, :]  # k=0, i=0,1
    W[1*2+0:1*2+2, 0:7] = W_2x2[2:4, :]  # k=1, i=0,1
    
    # Block 2: second 2 columns (k=2,3)
    U[:, 7:14] = U_2x2
    V[0*4+2:0*4+4, 7:14] = V_2x2[0:2, :]  # j=0, k=2,3
    V[1*4+2:1*4+4, 7:14] = V_2x2[2:4, :]  # j=1, k=2,3
    W[0*2+2:0*2+4, 7:14] = W_2x2[0:2, :]  # k=2, i=0,1
    W[1*2+2:1*2+4, 7:14] = W_2x2[2:4, :]  # k=3, i=0,1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try the deterministic Strassen-based rank-14 construction first
    result = _strassen_2x2x4()
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
        return result
    
    # Fallback: trivial rank 16 solution
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
