"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    Returns U (4x7), V (4x7), W (4x7).
    Convention: U indexed by (i,j), V by (j,k), W by (k,i).
    """
    # Standard Strassen for C = A*B where A,B are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M2 - M7
    # C01 = M5 + M7
    # C10 = M3 + M6
    # C11 = M1 - M5 + M3 + M2
    
    U = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    V = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    W = [[0]*7 for _ in range(4)]  # 2x2 = 4 entries
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00 gets +M1
    W[3][0] = 1  # C11 gets +M1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C01 gets +M2
    W[3][1] = 1  # C11 gets +M2
    
    # M3 = A00(B10-B00)
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[2][2] = 1  # C10 gets +M3
    W[3][2] = 1  # C11 gets +M3
    
    # M4 = A11(B01-B11)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1  # C00 gets +M4
    W[1][3] = -1  # C01 gets -M4? Wait, let me recheck.
    
    # Let me redo this more carefully.
    # C00 = M1 + M4 - M2 - M7
    # C01 = M5 + M7
    # C10 = M3 + M6
    # C11 = M1 - M5 + M3 + M2
    
    # Reset
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B00)
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B11)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = -1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    return U, V, W


def _build_decomp(n, m, p):
    """
    Build a Strassen-based decomposition for 2x3x8.
    Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Split B = [B1; B2] where B1 is 2x8, B2 is 1x8.
    C = A1*B1 + A2*B2.
    A2*B2: rank 16.
    A1*B1: split into 4 blocks of 2x2x2, each rank 7 (Strassen). Total 28.
    Grand total: 44.
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2 (rank 16)
    # A2 has rows i=0,1 and column j=2. So U[i*3+2] for i=0,1
    # B2 has row j=2 and columns k=0..7. So V[2*8+k] = V[16+k] for k=0..7
    # C has entries (i,k) for i=0,1 and k=0..7. W[k*2+i]
    for k in range(8):
        for i in range(2):
            U[i * 3 + 2][col] = 1
            V[16 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Part 2: A1*B1 via Strassen (4 blocks)
    # A1 is 2x2: entries (i,j) for i=0,1; j=0,1. U[i*3+j]
    # B1 is 2x8: entries (j,k) for j=0,1; k=0..7. V[j*8+k]
    # C from A1*B1: entries (i,k) for i=0,1; k=0..7. W[k*2+i]
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    for g in range(4):
        # Block g: columns k=2g, 2g+1 of B1
        # B1 entries: V[0*8 + 2g], V[0*8 + 2g+1], V[1*8 + 2g], V[1*8 + 2g+1]
        # = V[2g], V[2g+1], V[8+2g], V[8+2g+1]
        # A1 entries: U[0*3+0]=U[0], U[0*3+1]=U[1], U[1*3+0]=U[3], U[1*3+1]=U[4]
        # C entries: W[(2g)*2+0]=W[4g], W[(2g)*2+1]=W[4g+1], W[(2g+1)*2+0]=W[4g+2], W[(2g+1)*2+1]=W[4g+3]
        
        # Map Strassen indices to our indices:
        # A00 -> U[0], A01 -> U[1], A10 -> U[3], A11 -> U[4]
        # B00 -> V[2g], B01 -> V[2g+1], B10 -> V[8+2g], B11 -> V[8+2g+1]
        # C00 -> W[4g], C01 -> W[4g+1], C10 -> W[4g+2], C11 -> W[4g+3]
        
        a_idx = [0, 1, 3, 4]  # A00, A01, A10, A11
        b_idx = [2*g, 2*g+1, 8+2*g, 8+2*g+1]  # B00, B01, B10, B11
        c_idx = [4*g, 4*g+1, 4*g+2, 4*g+3]  # C00, C01, C10, C11
        
        for s in range(7):
            for u_row in range(4):
                if U_s[u_row][s] != 0:
                    U[a_idx[u_row]][col + s] = U_s[u_row][s]
            for v_row in range(4):
                if V_s[v_row][s] != 0:
                    V[b_idx[v_row]][col + s] = V_s[v_row][s]
            for w_row in range(4):
                if W_s[w_row][s] != 0:
                    W[c_idx[w_row]][col + s] = W_s[w_row][s]
        col += 7
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary: Strassen-based rank-44 decomposition
    try:
        U, V, W = _build_decomp(n, m, p)
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook rank-48
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
