import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks to achieve rank 56.
    The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        # Return a valid fallback (schoolbook, rank 60) if we're running out of time
        return _schoolbook_fallback(n, m, p)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # 1. Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: 0:C00, 1:C01, 2:C10, 3:C11
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        W[1, 0] = -1; W[2, 0] = -1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        W[0, 1] = 0; W[1, 1] = 0
        # Correction: C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
        # Let's use the standard Strassen formulas carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        W[1, 0] = 0; W[2, 0] = 0
        # C00 += M1, C11 += M1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        W[0, 1] = 0; W[1, 1] = 0
        # C10 += M2, C11 -= M2
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        W[0, 2] = 0; W[2, 2] = 0
        # C01 += M3, C11 += M3
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        W[1, 3] = 0; W[3, 3] = 0
        # C00 += M4, C10 += M4
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        W[2, 4] = 0; W[3, 4] = 0
        # C00 -= M5, C01 += M5
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        W[0, 5] = 0; W[1, 5] = 0; W[2, 5] = 0
        # C11 += M6
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = 0
        # C00 += M7
        
        return U, V, W
    
    # 2. Schoolbook 2x2x1 decomposition (rank 4, minimal)
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A00*B0 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A01*B1 -> C0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # Term 3: A10*B0 -> C1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # Term 4: A11*B1 -> C1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # 3. Schoolbook 1x2x2 decomposition (rank 4, minimal)
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A0*B00 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A1*B10 -> C0
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # Term 3: A0*B01 -> C1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        # Term 4: A1*B11 -> C1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    # 4. Schoolbook 1x2x1 decomposition (rank 2, minimal)
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        # Term 1: A0*B0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A1*B1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Assemble 3x4x5: split A into rows {0,1} and {2}, split B rows into {0,1} and {2,3}
    # U index = i*m+j, V index = j*p+k, W index = k*n+i
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4]
    
    for r in range(7):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10] # C1_top cols 2,3 -> 6,9. C1_bottom cols 2,3 -> 7,10.
    
    for r in range(7):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 3: A[0:2,0:2] * B[0:2,4:5] (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13] # C1_top col 4 -> 12. C1_bottom col 4 -> 13.
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 4: A[0:2,2:4] * B[2:4,0:2] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    
    for r in range(7):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 5: A[0:2,2:4] * B[2:4,2:4] (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    
    for r in range(7):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 6: A[0:2,2:4] * B[2:4,4:5] (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 7: A[2:3,0:2] * B[0:2,0:2] (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5] # C2 cols 0,1 -> 2,5.
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 8: A[2:3,0:2] * B[0:2,2:4] (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11] # C2 cols 2,3 -> 8,11.
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 9: A[2:3,0:2] * B[0:2,4:5] (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14] # C2 col 4 -> 14.
    
    for r in range(2):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 10: A[2:3,2:4] * B[2:4,0:2] (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 11: A[2:3,2:4] * B[2:4,2:4] (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    
    for r in range(4):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Block 12: A[2:3,2:4] * B[2:4,4:5] (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    
    for r in range(2):
        U_parts.append(np.zeros(n * m))
        for i, idx in enumerate(u_idx):
            U_parts[-1][idx] = U_blk[i, r]
        
        V_parts.append(np.zeros(m * p))
        for i, idx in enumerate(v_idx):
            V_parts[-1][idx] = V_blk[i, r]
        
        W_parts.append(np.zeros(p * n))
        for i, idx in enumerate(w_idx):
            W_parts[-1][idx] = W_blk[i, r]
    
    # Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook_fallback(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    U_parts = []
    V_parts = []
    W_parts = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1
                v = np.zeros(m * p)
                v[j * p + k] = 1
                w = np.zeros(p * n)
                w[k * n + i] = 1
                U_parts.append(u)
                V_parts.append(v)
                W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
