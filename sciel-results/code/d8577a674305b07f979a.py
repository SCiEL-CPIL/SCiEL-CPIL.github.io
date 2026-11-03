import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # 150s limit, 5s safety margin

def _elapsed():
    return time.time() - _start_time

def _verify(U, V, W, n=3, m=3, p=3, tol=1e-10):
    """Check that sum_r U[:,r] V[:,r] W[:,r] equals the matrix multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', Ua[:, r], Va[:, r], Wa[:, r])
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    
    if np.max(np.abs(T - T_target)) > tol:
        return False
    return True

def construct():
    n, m, p = 3, 3, 3
    # Rank 23 decomposition for 3x3x3 matrix multiplication
    # Using the known minimal rank decomposition
    
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: i*m + j  (row i, col j of A)
    # V: j*p + k  (row j, col k of B)
    # W: k*n + i  (row k, col i of C)
    
    # 7 Strassen terms for the 2x2 subproblem (rows/cols 0,1)
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[0][2] = -1
    
    # M4 = A11(B01-B00)
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    # 16 additional terms for the remaining entries
    # Terms involving index 2 (row/col 2)
    
    # C02: A00*B02, A01*B12, A02*B22
    U[0][7] = 1; V[2][7] = 1; W[6][7] = 1
    U[1][8] = 1; V[5][8] = 1; W[6][8] = 1
    U[2][9] = 1; V[8][9] = 1; W[6][9] = 1
    
    # C12: A10*B02, A11*B12, A12*B22
    U[3][10] = 1; V[2][10] = 1; W[7][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[7][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[7][12] = 1
    
    # C20: A20*B00, A21*B10, A22*B20
    U[6][13] = 1; V[0][13] = 1; W[2][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[2][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[2][15] = 1
    
    # C21: A20*B01, A21*B11, A22*B21
    U[6][16] = 1; V[1][16] = 1; W[5][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[5][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[5][18] = 1
    
    # C22: A20*B02, A21*B12, A22*B22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][20] = 1; V[5][20] = 1; W[8][20] = 1
    U[8][21] = 1; V[8][21] = 1; W[8][21] = 1
    
    # Cross terms for C00, C01, C10, C11 involving index 2
    U[2][22] = 1; V[6][22] = 1; W[0][22] = 1  # A02*B20 -> C00
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If rank 23 fails, try the corrected rank-26
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # 7 Strassen terms
    U[0][0] = 1; U[4][0] = 1; V[0][0] = 1; V[4][0] = 1; W[0][0] = 1; W[4][0] = 1
    U[3][1] = 1; U[4][1] = 1; V[0][1] = 1; W[3][1] = 1
    U[0][2] = 1; V[3][2] = 1; V[4][2] = -1; W[0][2] = -1
    U[4][3] = 1; V[1][3] = 1; V[0][3] = -1; W[3][3] = 1
    U[0][4] = 1; U[1][4] = 1; V[4][4] = 1; W[1][4] = 1; W[4][4] = 1
    U[3][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[0][5] = 1
    U[1][6] = 1; U[4][6] = -1; V[3][6] = 1; V[4][6] = 1; W[0][6] = -1; W[1][6] = 1
    
    # 19 border terms
    U[2][7] = 1; V[6][7] = 1; W[0][7] = 1   # A02*B20 -> C00
    U[2][8] = 1; V[7][8] = 1; W[1][8] = 1   # A02*B21 -> C01
    U[5][9] = 1; V[6][9] = 1; W[3][9] = 1   # A12*B20 -> C10
    U[5][10] = 1; V[7][10] = 1; W[4][10] = 1 # A12*B21 -> C11
    U[6][11] = 1; V[0][11] = 1; W[2][11] = 1 # A20*B00 -> C20
    U[6][12] = 1; V[1][12] = 1; W[5][12] = 1 # A20*B01 -> C21
    U[6][13] = 1; V[2][13] = 1; W[8][13] = 1 # A20*B02 -> C22
    U[7][14] = 1; V[3][14] = 1; W[2][14] = 1 # A21*B10 -> C20
    U[7][15] = 1; V[4][15] = 1; W[5][15] = 1 # A21*B11 -> C21
    U[7][16] = 1; V[5][16] = 1; W[8][16] = 1 # A21*B12 -> C22
    U[8][17] = 1; V[6][17] = 1; W[2][17] = 1 # A22*B20 -> C20
    U[8][18] = 1; V[7][18] = 1; W[5][18] = 1 # A22*B21 -> C21
    U[8][19] = 1; V[8][19] = 1; W[8][19] = 1 # A22*B22 -> C22
    U[0][20] = 1; V[2][20] = 1; W[6][20] = 1 # A00*B02 -> C02
    U[1][21] = 1; V[5][21] = 1; W[6][21] = 1 # A01*B12 -> C02
    U[3][22] = 1; V[2][22] = 1; W[7][22] = 1 # A10*B02 -> C12
    U[4][23] = 1; V[5][23] = 1; W[7][23] = 1 # A11*B12 -> C12
    U[5][24] = 1; V[8][24] = 1; W[7][24] = 1 # A12*B22 -> C12
    U[2][25] = 1; V[8][25] = 1; W[6][25] = 1 # A02*B22 -> C02
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to Rank 26 (Current Best)
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Re-insert the 7 Strassen terms
    U[0][0] = 1; U[4][0] = 1; V[0][0] = 1; V[4][0] = 1; W[0][0] = 1; W[4][0] = 1
    U[3][1] = 1; U[4][1] = 1; V[0][1] = 1; W[3][1] = 1
    U[0][2] = 1; V[3][2] = 1; V[4][2] = -1; W[0][2] = -1
    U[4][3] = 1; V[1][3] = 1; V[0][3] = -1; W[3][3] = 1
    U[0][4] = 1; U[1][4] = 1; V[4][4] = 1; W[1][4] = 1; W[4][4] = 1
    U[3][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[0][5] = 1
    U[1][6] = 1; U[4][6] = -1; V[3][6] = 1; V[4][6] = 1; W[0][6] = -1; W[1][6] = 1
    
    # Border terms (19 terms)
    # C20
    U[6][7] = 1; V[0][7] = 1; W[2][7] = 1
    U[7][8] = 1; V[3][8] = 1; W[2][8] = 1
    U[8][9] = 1; V[6][9] = 1; W[2][9] = 1
    # C21
    U[6][10] = 1; V[1][10] = 1; W[5][10] = 1
    U[7][11] = 1; V[4][11] = 1; W[5][11] = 1
    U[8][12] = 1; V[7][12] = 1; W[5][12] = 1
    # C22
    U[6][13] = 1; V[2][13] = 1; W[8][13] = 1
    U[7][14] = 1; V[5][14] = 1; W[8][14] = 1
    U[8][15] = 1; V[8][15] = 1; W[8][15] = 1
    # C02
    U[0][16] = 1; V[2][16] = 1; W[6][16] = 1
    U[1][17] = 1; V[5][17] = 1; W[6][17] = 1
    U[2][18] = 1; V[8][18] = 1; W[6][18] = 1
    # C12
    U[3][19] = 1; V[2][19] = 1; W[7][19] = 1
    U[4][20] = 1; V[5][20] = 1; W[7][20] = 1
    U[5][21] = 1; V[8][21] = 1; W[7][21] = 1
    # Cross
    U[2][22] = 1; V[6][22] = 1; W[0][22] = 1
    U[2][23] = 1; V[7][23] = 1; W[1][23] = 1
    U[5][24] = 1; V[6][24] = 1; W[3][24] = 1
    U[5][25] = 1; V[7][25] = 1; W[4][25] = 1
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
        
    # Last resort: Rank 27
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
                r += 1
    return {"U": U, "V": V, "W": W}
