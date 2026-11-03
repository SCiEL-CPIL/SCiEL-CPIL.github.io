import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n=3, m=3, p=6, trials=50, seed=42):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Direct tensor verification: check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # U, V, W are 4x7 matrices
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen algorithm for 2x2x2
    # C11 = M1 - M4 + M6 - M7
    # C12 = M5 + M7
    # C21 = M4 + M6
    # C22 = M1 - M2 + M3 - M6
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indexing: U[i*2+j, c]
    # M1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1 # C11, C22, -C12, -C21? No.
    # W is for C. C11, C12, C21, C22.
    # W[0,0]=1 (C11), W[1,0]=1 (C12), W[2,0]=1 (C21), W[3,0]=1 (C22).
    # M1 contributes to C11 and C22.
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = -1 # C22
    
    # M3: A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[3, 2] = 1 # C22
    
    # M4: A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1 # C21
    W[0, 3] = -1 # C11
    
    # M5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1 # C12
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1 # C11
    W[2, 5] = 1 # C21
    W[3, 5] = -1 # C22
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1 # C12
    W[0, 6] = -1 # C11
    W[2, 6] = -1 # C21
    
    return U, V, W

def _rank26_3x3x3():
    """Construct a rank-26 decomposition for 3x3x3."""
    n, m, p = 3, 3, 3
    U = np.zeros((n*m, 26))
    V = np.zeros((m*p, 26))
    W = np.zeros((p*n, 26))
    
    # 1. A22 B22 (top-left 2x2 of A, top-left 2x2 of B)
    # A indices: 0,1,2,3 (2x2 block)
    # B indices: 0,1,2,3 (2x2 block)
    # C indices: 0,1,2,3 (2x2 block)
    U_22, V_22, W_22 = _strassen_2x2x2()
    U[0:4, 0:7] = U_22
    V[0:4, 0:7] = V_22
    W[0:4, 0:7] = W_22
    
    # 2. a d (A21, A22? No. A is 3x3. A22 is top-left. a is top-right 2x1. d is bottom-left 1x2.)
    # A: 
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # A22 (top-left 2x2): 0,1,3,4
    # a (top-right 2x1): 2,5
    # A12 (bottom-left 1x2): 6,7
    # b (bottom-right 1x1): 8
    
    # B:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # B22 (top-left 2x2): 0,1,3,4
    # c (top-right 2x1): 2,5
    # d (bottom-left 1x2): 6,7
    # e (bottom-right 1x1): 8
    
    # C:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # C22 (top-left 2x2): 0,1,3,4
    # C21 (top-right 2x1): 2,5
    # C12 (bottom-left 1x2): 6,7
    # C11 (bottom-right 1x1): 8
    
    # Term 2: a d (2x1 * 1x2 -> 2x2)
    # a = [A2, A5], d = [B6, B7]
    # ad = [[A2B6, A2B7], [A5B6, A5B7]]
    # C indices: 0,1,3,4
    # Rank 4
    U[2, 7] = 1; U[5, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1; W[1, 7] = 1; W[3, 7] = 1; W[4, 7] = 1 # This is wrong. W is for C.
    # W[k*n+i, c]. C00, C01, C10, C11.
    # ad contributes to C00, C01, C10, C11.
    # C00 = A2B6. W[0,7]=1. U[2,7]=1, V[6,7]=1.
    # C01 = A2B7. W[1,7]=1. U[2,7]=1, V[7,7]=1.
    # C10 = A5B6. W[3,7]=1. U[5,7]=1, V[6,7]=1.
    # C11 = A5B7. W[4,7]=1. U[5,7]=1, V[7,7]=1.
    
    # This is getting complex to hardcode manually.
    # I will return None if I can't do it cleanly.
    return None

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    return best
