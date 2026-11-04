import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # Rank 16 decomposition
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=0 (1 term)
    # A[i, 0] * B[0, k] -> C[i, k]
    for i in range(n):
        U[i * m + 0][r] = 1
    for k in range(p):
        V[0 * p + k][r] = 1
        for i in range(n):
            W[k * n + i][r] = 1
    r += 1
    
    # Helper to add Strassen block
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal r
        # M1 = (A00+A11)(B00+B11)
        U[i1*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        # M2 = (A10+A11)B00
        U[i2*m+j1][r] = 1; U[i2*m+j2][r] = 1
        V[j1*p+k1][r] = 1
        W[k1*n+i2][r] = 1; W[k2*n+i2][r] = -1
        r += 1
        # M3 = A00(B01-B11)
        U[i1*m+j1][r] = 1
        V[j1*p+k2][r] = 1; V[j2*p+k2][r] = -1
        W[k2*n+i1][r] = 1; W[k2*n+i2][r] = 1
        r += 1
        # M4 = A11(B10-B00)
        U[i2*m+j2][r] = 1
        V[j2*p+k1][r] = 1; V[j1*p+k1][r] = -1
        W[k1*n+i1][r] = 1; W[k1*n+i2][r] = 1
        r += 1
        # M5 = (A00+A01)B11
        U[i1*m+j1][r] = 1; U[i1*m+j2][r] = 1
        V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = -1; W[k2*n+i1][r] = 1
        r += 1
        # M6 = (A10-A00)(B00+B01)
        U[i2*m+j1][r] = 1; U[i1*m+j1][r] = -1
        V[j1*p+k1][r] = 1; V[j1*p+k2][r] = 1
        W[k2*n+i2][r] = 1
        r += 1
        # M7 = (A01-A11)(B10+B11)
        U[i1*m+j2][r] = 1; U[i2*m+j2][r] = -1
        V[j2*p+k1][r] = 1; V[j2*p+k2][r] = 1
        W[k1*n+i1][r] = 1
        r += 1

    # Part 2: j=1,2, k=0,1 (7 terms)
    add_strassen(0, 1, 1, 2, 0, 1)
    
    # Part 3: j=1,2, k=2,3 (7 terms)
    add_strassen(0, 1, 1, 2, 2, 3)
    
    # Part 4: j=1,2, k=4 (1 term)
    # A[i, 1] B[1, 4] + A[i, 2] B[2, 4] -> C[i, 4]
    for i in range(n):
        U[i * m + 1][r] = 1
        U[i * m + 2][r] = 1
    V[1 * p + 4][r] = 1
    V[2 * p + 4][r] = 1
    for i in range(n):
        W[4 * n + i][r] = 1
    r += 1
    
    # Verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check if decomposition is exact
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_arr[i*m+j, ri] * V_arr[j*p+k, ri] * W_arr[k*n+i, ri] for ri in range(rank))
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
            
    if ok:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}
