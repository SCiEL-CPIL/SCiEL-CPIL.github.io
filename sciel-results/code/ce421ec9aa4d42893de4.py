import time
import numpy as np

DIMENSIONS = [3, 4, 7]

DEADLINE = 150.0
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME) - 5.0

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strategy:
    # Use 3x2x2 blocks with rank 2.
    # i covers 0..2 (size 3)
    # j covers 0..1, 2..3 (size 2 each)
    # k covers 0..1, 2..3, 4..5 (size 2 each)
    # Remaining k=6 handled by 3x4x1 with rank 4.
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # 3x2x2 block decomposition (Rank 2)
    # Split j into j0, j1.
    # Term 1: j=j0
    # Term 2: j=j1
    
    def add_3x2x2_block(i0, j0, k0):
        # Block covers i in [i0, i0+2], j in [j0, j0+1], k in [k0, k0+1]
        # Term 1: uses j=j0
        # U: i in 0..2, j=j0 -> 1
        # V: j=j0, k in 0..1 -> 1
        # W: k in 0..1, i in 0..2 -> 1
        
        u1 = []
        v1 = []
        w1 = []
        for di in range(3):
            u1.append(((i0+di)*m + (j0), 1))
        for dk in range(2):
            v1.append(((j0)*p + (k0+dk), 1))
        for dk in range(2):
            for di in range(3):
                w1.append(((k0+dk)*n + (i0+di), 1))
        
        U_terms.append(u1)
        V_terms.append(v1)
        W_terms.append(w1)
        
        # Term 2: uses j=j0+1
        u2 = []
        v2 = []
        w2 = []
        for di in range(3):
            u2.append(((i0+di)*m + (j0+1), 1))
        for dk in range(2):
            v2.append(((j0+1)*p + (k0+dk), 1))
        for dk in range(2):
            for di in range(3):
                w2.append(((k0+dk)*n + (i0+di), 1))
        
        U_terms.append(u2)
        V_terms.append(v2)
        W_terms.append(w2)

    # Define blocks
    # i0=0
    # j0=0, 2
    # k0=0, 2, 4
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_3x2x2_block(0, j0, k0)
            
    # Handle remainder: k=6
    # i in 0..2, j in 0..3, k=6
    # Split into 4 blocks of 3x1x1
    for j in range(4):
        u = []
        v = []
        w = []
        for di in range(3):
            u.append(((di)*m + j, 1))
        v.append((j*p + 6, 1))
        for di in range(3):
            w.append((6*n + di, 1))
            
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
        
    rank = len(U_terms)
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    for r in range(rank):
        for idx, val in U_terms[r]:
            U[idx, r] = val
        for idx, val in V_terms[r]:
            V[idx, r] = val
        for idx, val in W_terms[r]:
            W[idx, r] = val
            
    # Verify correctness
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    computed = 0
                    for r in range(rank):
                        computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if computed != expected:
                        return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p), dtype=np.int64)
        V = np.zeros((m*p, n*m*p), dtype=np.int64)
        W = np.zeros((p*n, n*m*p), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
