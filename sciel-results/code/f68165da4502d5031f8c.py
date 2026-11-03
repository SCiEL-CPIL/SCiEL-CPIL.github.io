import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Target rank
    R = 47
    
    # Initialize factor matrices
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen decomposition for 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A11-A01)(B01+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M5 - M1 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for 2x2 block
    # A is 3x4, B is 4x4, C is 4x3 (wait, C is p x n? No, C is n x p? 
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # W is p x n. So C is p x n?
    # Multiplication: A (n x m) * B (m x p) = C (n x p).
    # So C has indices i (0..n-1), k (0..p-1).
    # W index is k*n + i. So W[k*n+i] is C[i,k].
    
    # Let's map Strassen terms to U, V, W entries.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Term 0: M1
    U[0*m+0, 0] += 1  # A00
    U[1*m+1, 0] += 1  # A11
    V[0*p+0, 0] += 1  # B00
    V[1*p+1, 0] += 1  # B11
    W[0*n+0, 0] += 1  # C00
    W[1*n+1, 0] += 1  # C11
    
    # Term 1: M2
    U[1*m+0, 1] += 1  # A10
    U[1*m+1, 1] += 1  # A11
    V[0*p+0, 1] += 1  # B00
    W[0*n+1, 1] += 1  # C10
    
    # Term 2: M3
    U[0*m+0, 2] += 1  # A00
    V[1*p+0, 2] += 1  # B10
    V[1*p+1, 2] += -1 # B11
    W[0*n+0, 2] += 1  # C00
    W[1*n+0, 2] += 1  # C10? No, C01 is i=0, k=1. W[1*n+0].
    # C01 = M3 + M4 + M7. So M3 contributes to C01.
    # C01 index: i=0, k=1. W[1*n+0].
    W[1*n+0, 2] += 1
    
    # Term 3: M4
    U[1*m+1, 3] += 1  # A11
    V[0*p+1, 3] += 1  # B01
    V[0*p+0, 3] += -1 # B00
    W[0*n+0, 3] += -1 # C00
    W[1*n+0, 3] += 1  # C01
    
    # Term 4: M5
    U[0*m+0, 4] += 1  # A00
    U[0*m+1, 4] += 1  # A01? No, A00+A10. j=0,1.
    # A is n x m. A00 is i=0,j=0. A10 is i=1,j=0.
    # So U[0*m+0] and U[1*m+0].
    U[1*m+0, 4] += 1
    V[1*p+1, 4] += 1  # B11
    W[0*n+0, 4] += 1  # C00
    W[0*n+1, 4] += 1  # C10
    
    # Term 5: M6
    U[1*m+0, 5] += 1  # A10
    U[0*m+0, 5] += -1 # A00
    V[0*p+0, 5] += 1  # B00
    V[1*p+0, 5] += 1  # B10
    W[0*n+1, 5] += 1  # C10
    W[1*n+1, 5] += 1  # C11
    
    # Term 6: M7
    U[1*m+1, 6] += 1  # A11
    U[0*m+1, 6] += -1 # A01
    V[0*p+1, 6] += 1  # B01
    V[1*p+1, 6] += 1  # B11
    W[0*n+0, 6] += -1 # C00
    W[1*n+0, 6] += 1  # C01
    W[1*n+1, 6] += 1  # C11
    
    # Fill remaining terms with schoolbook
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if in 2x2x2 block
                if i < 2 and j < 2 and k < 2:
                    continue
                
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Verify
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

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
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
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
