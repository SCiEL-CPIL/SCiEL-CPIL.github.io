import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _build_rank11_2x2x3():
    """
    Build a rank-11 decomposition for 2x2x3 matrix multiplication.
    A is 2x2, B is 2x3, C is 2x3.
    Uses Strassen (rank 7) for the first 2 columns of B, and schoolbook (rank 4) for the 3rd column.
    """
    # U: (2*2, 11) = (4, 11)
    # V: (2*3, 11) = (6, 11)
    # W: (3*2, 11) = (6, 11)
    U = np.zeros((4, 11), dtype=np.float64)
    V = np.zeros((6, 11), dtype=np.float64)
    W = np.zeros((6, 11), dtype=np.float64)
    
    # Strassen for 2x2x2 (columns 0, 1 of B, which are indices 0,1 and 3,4 in V)
    # B indices: j*3+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # j=0, k=2 -> 2
    # j=1, k=2 -> 5
    
    # Strassen terms for A * B[:, 0:2]
    # M1 = (a+d)(e+i) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)(e) -> C10, C11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # Wait, Strassen W indices depend on output C.
    # Let's use standard Strassen mapping.
    # C = [[C00, C01], [C10, C11]]
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
    # M3 = (A00)(B10-B11) -> C01 += M3
    # M4 = (A11)(B01-B00) -> C10 -= M4
    # M5 = (A00+A01)(B10+B11) -> C01 += M5, C11 += M5
    # M6 = (A10-A00)(B00+B01) -> C10 += M6, C00 -= M6
    # M7 = (A01-A11)(B10+B11) -> C01 += M7, C11 -= M7
    
    # Reset and re-apply carefully.
    U = np.zeros((4, 11), dtype=np.float64)
    V = np.zeros((6, 11), dtype=np.float64)
    W = np.zeros((6, 11), dtype=np.float64)
    
    # Term 0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1 # B00, B11
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # Term 1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1 # B00
    W[2, 1] = 1; W[3, 1] = -1 # C10, C11
    
    # Term 2: M3
    U[0, 2] = 1
    V[3, 2] = 1; V[4, 2] = -1 # B10, B11
    W[1, 2] = 1 # C01
    
    # Term 3: M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1 # B01, B00
    W[2, 3] = -1 # C10
    
    # Term 4: M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1; V[4, 4] = 1 # B10, B11
    W[1, 4] = 1; W[3, 4] = 1 # C01, C11
    
    # Term 5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1 # B00, B01
    W[2, 5] = 1; W[0, 5] = -1 # C10, C00
    
    # Term 6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1 # B10, B11
    W[1, 6] = 1; W[3, 6] = -1 # C01, C11
    
    # Terms 7-10: Schoolbook for column 2 (k=2)
    # B indices: j=0, k=2 -> 2. j=1, k=2 -> 5.
    # C indices: k=2, i=0 -> 4. k=2, i=1 -> 5.
    
    # M7 = A00 * B02
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1
    
    # M8 = A01 * B12
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1
    
    # M9 = A10 * B02
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # M10 = A11 * B12
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Build 4 blocks of 2x2x3
    U_223, V_223, W_223 = _build_rank11_2x2x3()
    rank_223 = 11
    total_rank = 4 * rank_223
    
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:3] -> C[0:2, 0:3]
    base = 0
    for c in range(rank_223):
        col = base + c
        for i in range(2):
            for j_local in range(2):
                U[i*4 + j_local, col] = U_223[i*2+j_local, c]
        for j_local in range(2):
            for k_local in range(3):
                V[j_local*6 + k_local, col] = V_223[j_local*3+k_local, c]
        for k_local in range(3):
            for i in range(2):
                W[k_local*2 + i, col] = W_223[k_local*2+i, c]
                
    # Block (0,1): A[0:2, 0:2] * B[0:2, 3:6] -> C[0:2, 3:6]
    base = rank_223
    for c in range(rank_223):
        col = base + c
        for i in range(2):
            for j_local in range(2):
                U[i*4 + j_local, col] = U_223[i*2+j_local, c]
        for j_local in range(2):
            for k_local in range(3):
                V[j_local*6 + (3+k_local), col] = V_223[j_local*3+k_local, c]
        for k_local in range(3):
            for i in range(2):
                W[(3+k_local)*2 + i, col] = W_223[k_local*2+i, c]
                
    # Block (1,0): A[0:2, 2:4] * B[2:4, 0:3] -> C[0:2, 0:3]
    base = 2 * rank_223
    for c in range(rank_223):
        col = base + c
        for i in range(2):
            for j_local in range(2):
                U[i*4 + (2+j_local), col] = U_223[i*2+j_local, c]
        for j_local in range(2):
            for k_local in range(3):
                V[(2+j_local)*6 + k_local, col] = V_223[j_local*3+k_local, c]
        for k_local in range(3):
            for i in range(2):
                W[k_local*2 + i, col] = W_223[k_local*2+i, c]
                
    # Block (1,1): A[0:2, 2:4] * B[2:4, 3:6] -> C[0:2, 3:6]
    base = 3 * rank_223
    for c in range(rank_223):
        col = base + c
        for i in range(2):
            for j_local in range(2):
                U[i*4 + (2+j_local), col] = U_223[i*2+j_local, c]
        for j_local in range(2):
            for k_local in range(3):
                V[(2+j_local)*6 + (3+k_local), col] = V_223[j_local*3+k_local, c]
        for k_local in range(3):
            for i in range(2):
                W[(3+k_local)*2 + i, col] = W_223[k_local*2+i, c]
                
    # Verify
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    
    if not np.allclose(recon, target, atol=1e-9):
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
