import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _strassen_222():
    # Returns U, V, W for 2x2x2 Strassen
    # C = AB
    # U maps A, V maps B, W maps C
    # Indices: A[i,j], B[j,k], C[i,k]
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # A is 2x2, B is 2x2, C is 2x2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U entries for A
    # A00: M1, M3, M5, M6, M7
    # A11: M1, M2, M4, M5, M7
    # A10: M2, M6
    # A01: M3, M5, M7
    
    # V entries for B
    # B00: M1, M2, M4, M6
    # B11: M1, M3, M5, M7
    # B01: M3, M5, M6, M7
    # B10: M4, M6, M7
    
    # W entries for C
    # C00: M1, M4, M5, M7
    # C01: M3, M5
    # C10: M2, M4
    # C11: M1, M2, M3, M6
    
    # Col 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Col 1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Col 2: M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Col 3: M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Col 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Col 5: M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Col 6: M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _combine(U1, V1, W1, U2, V2, W2, offset_u, offset_v, offset_w, offset_c):
    # This is not quite right for combining tensors directly into global U,V,W
    # We need to build the global matrices directly.
    pass

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition recursively
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We accumulate columns
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(i_start, i_end, j_start, j_end, k_start, k_end, U_block, V_block, W_block):
        # U_block is (ni*mj, rb), V_block is (mj*pk, rb), W_block is (pk*ni, rb)
        # We need to map local indices to global indices
        ni = i_end - i_start
        mj = j_end - j_start
        pk = k_end - k_start
        rb = len(U_block[0]) if U_block else 0
        
        U_g = [[0]*rb for _ in range(n*m)]
        V_g = [[0]*rb for _ in range(m*p)]
        W_g = [[0]*rb for _ in range(p*n)]
        
        for li in range(ni):
            for lj in range(mj):
                for c in range(rb):
                    U_g[(i_start+li)*m + (j_start+lj)][c] = U_block[li*mj+lj][c]
                    
        for lj in range(mj):
            for lk in range(pk):
                for c in range(rb):
                    V_g[(j_start+lj)*p + (k_start+lk)][c] = V_block[lj*pk+lk][c]
                    
        for lk in range(pk):
            for li in range(ni):
                for c in range(rb):
                    W_g[(k_start+lk)*n + (i_start+li)][c] = W_block[lk*ni+li][c]
                    
        U_cols.append(U_g)
        V_cols.append(V_g)
        W_cols.append(W_g)
        
    def get_decomp(n, m, p):
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        if n == 1:
            # Schoolbook
            rank = m * p
            U = [[0]*rank for _ in range(m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p)]
            col = 0
            for j in range(m):
                for k in range(p):
                    U[j][col] = 1
                    V[j*p+k][col] = 1
                    W[k][col] = 1
                    col += 1
            return U, V, W
        if m == 1:
            rank = n * p
            U = [[0]*rank for _ in range(n)]
            V = [[0]*rank for _ in range(p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i][col] = 1
                    V[k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
            return U, V, W
        if p == 1:
            rank = n * m
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m)]
            W = [[0]*rank for _ in range(n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    U[i*m+j][col] = 1
                    V[j][col] = 1
                    W[i][col] = 1
                    col += 1
            return U, V, W
            
        if n == 2 and m == 2 and p == 2:
            return _strassen_222()
            
        # Recursive partition
        # Partition the largest dimension
        if n >= m and n >= p:
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = get_decomp(n1, m, p)
            U2, V2, W2 = get_decomp(n2, m, p)
            # Combine
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            rank = rank1 + rank2
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            for i in range(n1):
                for j in range(m):
                    for c in range(rank1):
                        U[i*m+j][c] = U1[i*m+j][c]
            for i in range(n2):
                for j in range(m):
                    for c in range(rank2):
                        U[(n1+i)*m+j][rank1+c] = U2[i*m+j][c]
                        
            for j in range(m):
                for k in range(p):
                    for c in range(rank1):
                        V[j*p+k][c] = V1[j*p+k][c]
                    for c in range(rank2):
                        V[j*p+k][rank1+c] = V2[j*p+k][c]
                        
            for k in range(p):
                for i in range(n1):
                    for c in range(rank1):
                        W[k*n+i][c] = W1[k*n1+i][c]
                for i in range(n2):
                    for c in range(rank2):
                        W[k*n+(n1+i)][rank1+c] = W2[k*n2+i][c]
                        
            return U, V, W
            
        elif m >= n and m >= p:
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = get_decomp(n, m1, p)
            U2, V2, W2 = get_decomp(n, m2, p)
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            rank = rank1 + rank2
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            for i in range(n):
                for j in range(m1):
                    for c in range(rank1):
                        U[i*m+j][c] = U1[i*m1+j][c]
                for j in range(m2):
                    for c in range(rank2):
                        U[i*m+(m1+j)][rank1+c] = U2[i*m2+j][c]
                        
            for j in range(m1):
                for k in range(p):
                    for c in range(rank1):
                        V[j*p+k][c] = V1[j*p+k][c]
                for j in range(m2):
                    for c in range(rank2):
                        V[(m1+j)*p+k][rank1+c] = V2[j*p+k][c]
                        
            for k in range(p):
                for i in range(n):
                    for c in range(rank1):
                        W[k*n+i][c] = W1[k*n+i][c]
                    for c in range(rank2):
                        W[k*n+i][rank1+c] = W2[k*n+i][c]
                        
            return U, V, W
            
        else: # p is largest
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = get_decomp(n, m, p1)
            U2, V2, W2 = get_decomp(n, m, p2)
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            rank = rank1 + rank2
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            for i in range(n):
                for j in range(m):
                    for c in range(rank1):
                        U[i*m+j][c] = U1[i*m+j][c]
                    for c in range(rank2):
                        U[i*m+j][rank1+c] = U2[i*m+j][c]
                        
            for j in range(m):
                for k in range(p1):
                    for c in range(rank1):
                        V[j*p+k][c] = V1[j*p1+k][c]
                for k in range(p2):
                    for c in range(rank2):
                        V[j*p+(p1+k)][rank1+c] = V2[j*p2+k][c]
                        
            for k in range(p1):
                for i in range(n):
                    for c in range(rank1):
                        W[k*n+i][c] = W1[k*n+i][c]
                for k in range(p2):
                    for c in range(rank2):
                        W[(p1+k)*n+i][rank1+c] = W2[k*n+i][c]
                        
            return U, V, W

    U, V, W = get_decomp(n, m, p)
    
    # Convert to lists of lists if needed, but numpy arrays are fine?
    # The task says "integer/half-integer entries".
    # Strassen has -1, 1.
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback to schoolbook if timeout? No, get_decomp is fast.
        pass
        
    return {"U": U, "V": V, "W": W}
