DIMENSIONS = [2, 2, 4]
import numpy as np
import time
import random

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

def _trivial_solution(n, m, p):
    """Return the trivial rank n*m*p solution."""
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

def _strassen_2x2x4():
    """
    Construct a rank-14 tensor decomposition for 2x2 * 2x4 matrix multiplication.
    
    We use two Strassen blocks.
    Block 1 handles C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]
    Block 2 handles C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4]
    
    Indices:
    U rows: i*m+j (i in 0..1, j in 0..1) -> 0,1,2,3
    V rows: j*p+k (j in 0..1, k in 0..3) -> 0,1,2,3, 4,5,6,7
    W rows: k*n+i (k in 0..3, i in 0..1) -> 0,1, 2,3, 4,5, 6,7
    
    Strassen formulas for 2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    We need to map these to the global indices.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: k in {0,1}
    # r indices 0..6
    # M1: (A00+A11)(B00+B11)
    # U: A00 (row 0), A11 (row 3)
    # V: B00 (row 0), B11 (row 2)
    # W: C00 (W[0*2+0]=W[0]), C11 (W[1*2+1]=W[3])
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    # U: A10 (row 2), A11 (row 3)
    # V: B00 (row 0)
    # W: C10 (W[0*2+1]=W[1]), C11 (W[1*2+1]=W[3])
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A00(B10-B11)
    # U: A00 (row 0)
    # V: B10 (row 1), B11 (row 2)
    # W: C00 (W[0]), C01 (W[1*2+0]=W[2])
    U[0][2] = 1
    V[1][2] = 1; V[2][2] = -1
    W[0][2] = 1; W[2][2] = 1
    
    # M4: A11(B01-B00)
    # U: A11 (row 3)
    # V: B01 (row 1), B00 (row 0)
    # W: C00 (W[0]), C10 (W[1])
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A00+A01)B11
    # U: A00 (row 0), A01 (row 1)
    # V: B11 (row 2)
    # W: C00 (W[0]), C01 (W[2])
    U[0][4] = 1; U[1][4] = 1
    V[2][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # U: A10 (row 2), A00 (row 0)
    # V: B00 (row 0), B01 (row 1)
    # W: C11 (W[3])
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # U: A01 (row 1), A11 (row 3)
    # V: B10 (row 1), B11 (row 2)
    # W: C00 (W[0])
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[2][6] = 1
    W[0][6] = 1

    # Block 2: k in {2,3}
    # r indices 7..13
    # Same structure, but V indices shifted by 4 (k=2,3)
    # W indices shifted by 4 (k=2,3)
    
    # M1': (A00+A11)(B20+B31)
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[6][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # M2': (A10+A11)B20
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # M3': A00(B30-B31)
    U[0][9] = 1
    V[5][9] = 1; V[6][9] = -1
    W[4][9] = 1; W[6][9] = 1
    
    # M4': A11(B21-B20)
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # M5': (A00+A01)B31
    U[0][11] = 1; U[1][11] = 1
    V[6][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # M6': (A10-A00)(B20+B21)
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[5][12] = 1
    W[7][12] = 1
    
    # M7': (A01-A11)(B30+B31)
    U[1][13] = 1; U[3][13] = -1
    V[5][13] = 1; V[6][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Try the structured rank-14 solution first
    sol = _strassen_2x2x4()
    
    # Verify it
    if _verify_decomposition(sol['U'], sol['V'], sol['W'], n, m, p, tol=1e-9):
        return sol
    
    # Fallback to trivial solution if verification fails (should not happen)
    return _trivial_solution(n, m, p)
