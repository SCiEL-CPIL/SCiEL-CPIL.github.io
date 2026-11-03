import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    def verify(U, V, W, tol=1e-8):
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k2*n+i2, r]
                            exp = 1.0 if (i == i2 and k == k2) else 0.0
                            if abs(s - exp) > tol:
                                return False
        return True

    # Schoolbook fallback
    def schoolbook():
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return U_sb, V_sb, W_sb

    # Try rank-44 construction: 2x4x4 Strassen + 1x4x4 schoolbook
    # For 2x4x4: use 11 terms (known optimal for 2x4x4)
    # For 1x4x4: use 16 terms
    # Total: 27? No, the W factors couple them.
    # Actually, the total rank is the sum since the U factors for row 2 don't overlap.
    # But W factors do overlap, so we need to be careful.
    # Let's just try: 7 terms for 2x2x2, then optimize the rest.
    
    # Start with rank 47 (current approach) and verify
    R = 47
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    # Using standard Strassen formulas
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0*m+0, 0] += 1; U[1*m+1, 0] += 1
    V[0*p+0, 0] += 1; V[1*p+1, 0] += 1
    W[0*n+0, 0] += 1; W[1*n+1, 0] += 1
    
    # M2 = (A10+A11)B00 -> C10 += M2
    U[1*m+0, 1] += 1; U[1*m+1, 1] += 1
    V[0*p+0, 1] += 1
    W[0*n+1, 1] += 1
    
    # M3 = A00(B10-B11) -> C01 += M3
    U[0*m+0, 2] += 1
    V[1*p+0, 2] += 1; V[1*p+1, 2] += -1
    W[1*n+0, 2] += 1
    
    # M4 = A11(B01-B00) -> C00 -= M4, C01 += M4
    U[1*m+1, 3] += 1
    V[0*p+1, 3] += 1; V[0*p+0, 3] += -1
    W[0*n+0, 3] += -1; W[1*n+0, 3] += 1
    
    # M5 = (A00+A10)B11 -> C00 += M5, C10 += M5
    U[0*m+0, 4] += 1; U[1*m+0, 4] += 1
    V[1*p+1, 4] += 1
    W[0*n+0, 4] += 1; W[0*n+1, 4] += 1
    
    # M6 = (A10-A00)(B00+B10) -> C10 += M6, C11 += M6
    U[1*m+0, 5] += 1; U[0*m+0, 5] += -1
    V[0*p+0, 5] += 1; V[1*p+0, 5] += 1
    W[0*n+1, 5] += 1; W[1*n+1, 5] += 1
    
    # M7 = (A11-A01)(B01+B11) -> C00 -= M7, C01 += M7, C11 += M7
    U[1*m+1, 6] += 1; U[0*m+1, 6] += -1
    V[0*p+1, 6] += 1; V[1*p+1, 6] += 1
    W[0*n+0, 6] += -1; W[1*n+0, 6] += 1; W[1*n+1, 6] += 1
    
    # Schoolbook for remaining
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U_sb, V_sb, W_sb = schoolbook()
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
