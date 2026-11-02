import numpy as np
import time

def _strassen_2x2():
    # Returns U, V, W for 2x2 Strassen (Rank 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    # U, V, W shapes: (4, 7)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1 # C10 += M2, C11 -= M2?
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01 += M3, C11 += M3
    
    # M4 = A11(B01 - B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00 += M4
    
    # M5 = (A00 + A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # C00 -= M5, C10 += M5
    
    # M6 = (A01 - A00)(B00 + B10)
    U[1, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1 # C11 += M6
    
    # M7 = (A11 - A01)(B01 + B11)
    U[3, 6] = 1; U[1, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1 # C00 -= M7, C01 += M7
    
    return U, V, W

def _schoolbook_3x3():
    # Returns U, V, W for 3x3 Schoolbook (Rank 27)
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3 + j, r] = 1
                V[j*3 + k, r] = 1
                W[k*3 + i, r] = 1
                r += 1
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 110.0
    
    # Build 189 rank decomposition
    U3, V3, W3 = _schoolbook_3x3()
    U2, V2, W2 = _strassen_2x2()
    
    R3 = 27
    R2 = 7
    R_total = R3 * R2
    
    U = np.zeros((36, R_total))
    V = np.zeros((36, R_total))
    W = np.zeros((36, R_total))
    
    for r3 in range(R3):
        for r2 in range(R2):
            r = r3 * R2 + r2
            for i in range(3):
                for a in range(2):
                    idx = i * 2 + a
                    U[idx, r] = U3[i, r3] * U2[a, r2]
            for j in range(3):
                for b in range(2):
                    idx = j * 2 + b
                    V[idx, r] = V3[j, r3] * V2[b, r2]
            for k in range(3):
                for c in range(2):
                    idx = k * 2 + c
                    W[idx, r] = W3[k, r3] * W2[c, r2]
                    
    # Verify
    if _verify(6, 6, 6, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook 216
        U_sb = np.zeros((36, 216))
        V_sb = np.zeros((36, 216))
        W_sb = np.zeros((36, 216))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U_sb[i*6+j, col] = 1
                    V_sb[j*6+k, col] = 1
                    W_sb[k*6+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def _verify(n, m, p, U, V, W, tol=1e-8):
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check some zero entries
    np.random.seed(42)
    for _ in range(100):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i1 = a // m; j1 = a % m
        j2 = b // p; k1 = b % p
        k2 = c // n; i2 = c % n
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True
