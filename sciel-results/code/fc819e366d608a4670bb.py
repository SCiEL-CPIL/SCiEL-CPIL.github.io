import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# Memoization for rank calculation
rank_cache = {}

def get_min_rank(n, m, p):
    key = (n, m, p)
    if key in rank_cache:
        return rank_cache[key]
    
    # Base cases
    if n == 1 and m == 1 and p == 1:
        rank_cache[key] = 1
        return 1
    if n == 1 or m == 1 or p == 1:
        rank_cache[key] = n * m * p
        return n * m * p
    if n == 2 and m == 2 and p == 2:
        rank_cache[key] = 7
        return 7
        
    best = n * m * p
    
    # Try splitting n
    if n >= 2:
        n1, n2 = 2, n - 2
        r1 = get_min_rank(n1, m, p)
        r2 = get_min_rank(n2, m, p)
        best = min(best, r1 + r2)
        
    # Try splitting m
    if m >= 2:
        m1, m2 = 2, m - 2
        r1 = get_min_rank(n, m1, p)
        r2 = get_min_rank(n, m2, p)
        best = min(best, r1 + r2)
        
    # Try splitting p
    if p >= 2:
        p1, p2 = 2, p - 2
        r1 = get_min_rank(n, m, p1)
        r2 = get_min_rank(n, m, p2)
        best = min(best, r1 + r2)
        
    rank_cache[key] = best
    return best

def construct_strassen_222():
    # Returns U, V, W for 2x2x2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen terms
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to express these as U, V, W vectors
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the 7 multiplications
    # k=0: M1
    U[:, 0] = [1, 0, 1, 1] # A11, A12, A21, A22 ? No.
    # U rows are A11, A12, A21, A22.
    # A11 is row 0, A12 row 1, A21 row 2, A22 row 3.
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1 # B11, B12, B21, B22. B11 row 0, B12 row 1, B21 row 2, B22 row 3.
    # W: C11, C12, C21, C22.
    # C11 gets M1. C22 gets M1.
    W[0, 0] = 1; W[3, 0] = 1
    
    # k=1: M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21
    
    # k=2: M3 = A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C12, C22
    
    # k=3: M4 = A22(B12-B11)
    U[3, 3] = 1
    V[0, 3] = -1; V[1, 3] = 1
    W[0, 3] = -1; W[2, 3] = 1 # C11, C21
    
    # k=4: M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1 # C11, C12
    
    # k=5: M6 = (A21-A11)(B11+B12)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22
    
    # k=6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1 # C11
    
    return U, V, W

def construct_decomp(n, m, p):
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W
        
    if n == 2 and m == 2 and p == 2:
        return construct_strassen_222()
        
    if n == 1 or m == 1 or p == 1:
        # Schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Determine best split
    best_split = None
    best_rank = n * m * p
    
    # Try n split
    if n >= 2:
        n1, n2 = 2, n - 2
        r1 = get_min_rank(n1, m, p)
        r2 = get_min_rank(n2, m, p)
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            best_split = ('n', n1, n2)
            
    # Try m split
    if m >= 2:
        m1, m2 = 2, m - 2
        r1 = get_min_rank(n, m1, p)
        r2 = get_min_rank(n, m2, p)
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            best_split = ('m', m1, m2)
            
    # Try p split
    if p >= 2:
        p1, p2 = 2, p - 2
        r1 = get_min_rank(n, m, p1)
        r2 = get_min_rank(n, m, p2)
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            best_split = ('p', p1, p2)
            
    if best_split is None:
        # Fallback to schoolbook if no split found (should not happen for n,m,p > 1)
        return construct_decomp(n, m, p) # Recursion error? No, just use schoolbook
        # Actually if n,m,p > 1, we can always split 2.
        
    axis, d1, d2 = best_split
    
    if axis == 'n':
        U1, V1, W1 = construct_decomp(d1, m, p)
        U2, V2, W2 = construct_decomp(d2, m, p)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        rank = r1 + r2
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # U: Stack vertically
        U[:d1 * m, :r1] = U1
        U[d1 * m:, r1:] = U2
        
        # V: Concatenate horizontally
        V[:, :r1] = V1
        V[:, r1:] = V2
        
        # W: Place in correct rows
        # W1 rows correspond to i in 0..d1-1
        # W2 rows correspond to i in d1..n-1
        for k in range(p):
            for i1 in range(d1):
                W[k * n + i1, :r1] = W1[k * d1 + i1, :]
            for i2 in range(d2):
                W[k * n + d1 + i2, r1:] = W2[k * d2 + i2, :]
                
        return U, V, W
        
    elif axis == 'm':
        U1, V1, W1 = construct_decomp(n, d1, p)
        U2, V2, W2 = construct_decomp(n, d2, p)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        rank = r1 + r2
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # U: Place in correct rows
        # U1 rows j in 0..d1-1
        # U2 rows j in d1..m-1
        for i in range(n):
            for j1 in range(d1):
                U[i * m + j1, :r1] = U1[i * d1 + j1, :]
            for j2 in range(d2):
                U[i * m + d1 + j2, r1:] = U2[i * d2 + j2, :]
                
        # V: Stack vertically
        # V1 rows j in 0..d1-1
        # V2 rows j in d1..m-1
        V[:d1 * p, :r1] = V1
        V[d1 * p:, r1:] = V2
        
        # W: Concatenate horizontally
        W[:, :r1] = W1
        W[:, r1:] = W2
        
        return U, V, W
        
    elif axis == 'p':
        U1, V1, W1 = construct_decomp(n, m, d1)
        U2, V2, W2 = construct_decomp(n, m, d2)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        rank = r1 + r2
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # U: Concatenate horizontally
        U[:, :r1] = U1
        U[:, r1:] = U2
        
        # V: Place in correct cols
        # V1 cols k in 0..d1-1
        # V2 cols k in d1..p-1
        for j in range(m):
            for k1 in range(d1):
                V[j * p + k1, :r1] = V1[j * d1 + k1, :]
            for k2 in range(d2):
                V[j * p + d1 + k2, r1:] = V2[j * d2 + k2, :]
                
        # W: Stack vertically
        # W1 rows k in 0..d1-1
        # W2 rows k in d1..p-1
        W[:d1 * n, :r1] = W1
        W[d1 * n:, r1:] = W2
        
        return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Get the optimal decomposition
    U, V, W = construct_decomp(n, m, p)
    
    # Convert to list of lists for the required format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Check time
    if time.time() - start_time > 140:
        # Fallback to schoolbook if it takes too long (should not happen)
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U_list, "V": V_list, "W": W_list}
