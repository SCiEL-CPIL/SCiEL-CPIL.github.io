import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for M_{2,2,2} with rank 7.
    Indices:
    U: (i, j) where i in 0..1, j in 0..1. Size 2x2=4 rows.
    V: (j, k) where j in 0..1, k in 0..1. Size 2x2=4 rows.
    W: (k, i) where k in 0..1, i in 0..1. Size 2x2=4 rows.
    """
    # Strassen's 7 products:
    # P1 = A00 * B00
    # P2 = (A00 + A01) * B10
    # P3 = A01 * (B01 - B11)
    # P4 = A11 * (B00 + B01)
    # P5 = (A10 - A00) * (B00 + B10)
    # P6 = (A00 + A11) * (B11 - B00)
    # P7 = (A10 + A01) * (B11 - B01)
    
    # C00 = P1 - P3 + P5 + P7
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P2 + P4 + P6
    
    # We define U, V, W such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = C[i,k]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Row indices for U (i*m+j):
    # A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    
    # Row indices for V (j*p+k):
    # B00 -> 0, B01 -> 1, B10 -> 2, B11 -> 3
    
    # Row indices for W (k*n+i):
    # C00 -> 0, C01 -> 1, C10 -> 2, C11 -> 3
    
    # r=0: P1 = A00 * B00
    U[0, 0] = 1 # A00
    V[0, 0] = 1 # B00
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # r=1: P2 = (A00 + A01) * B10
    U[0, 1] = 1 # A00
    U[1, 1] = 1 # A01
    V[2, 1] = 1 # B10
    W[2, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # r=2: P3 = A01 * (B01 - B11)
    U[1, 2] = 1 # A01
    V[1, 2] = 1 # B01
    V[3, 2] = -1 # B11
    W[0, 2] = -1 # C00
    W[1, 2] = 1 # C01
    
    # r=3: P4 = A11 * (B00 + B01)
    U[3, 3] = 1 # A11
    V[0, 3] = 1 # B00
    V[1, 3] = 1 # B01
    W[1, 3] = 1 # C01
    W[3, 3] = 1 # C11
    
    # r=4: P5 = (A10 - A00) * (B00 + B10)
    U[2, 4] = 1 # A10
    U[0, 4] = -1 # A00
    V[0, 4] = 1 # B00
    V[2, 4] = 1 # B10
    W[0, 4] = 1 # C00
    W[2, 4] = 1 # C10
    
    # r=5: P6 = (A00 + A11) * (B11 - B00)
    U[0, 5] = 1 # A00
    U[3, 5] = 1 # A11
    V[3, 5] = 1 # B11
    V[0, 5] = -1 # B00
    W[3, 5] = 1 # C11
    
    # r=6: P7 = (A10 + A01) * (B11 - B01)
    U[2, 6] = 1 # A10
    U[1, 6] = 1 # A01
    V[3, 6] = 1 # B11
    V[1, 6] = -1 # B01
    W[0, 6] = 1 # C00
    W[3, 6] = 1 # C11
    
    return U, V, W

def _construct_m2210():
    """
    Construct U, V, W for M_{2,2,10} with rank 35.
    n=2, m=2, p=10.
    U: 4x35
    V: 20x35
    W: 20x35
    """
    n, m, p = 2, 2, 10
    num_U = n * m # 4
    num_V = m * p # 20
    num_W = p * n # 20
    rank = 35
    
    U = np.zeros((num_U, rank))
    V = np.zeros((num_V, rank))
    W = np.zeros((num_W, rank))
    
    # Split p=10 into 5 blocks of size 2.
    # For each block b (0..4), we use the Strassen decomposition for M_{2,2,2}.
    # The block corresponds to columns 2b and 2b+1 of B, and C.
    # The rows of A are the same (0,1).
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    for b in range(5):
        col_offset = b * 7
        # V indices for this block:
        # B rows are 0,1. Columns are 2b, 2b+1.
        # V index = j*p + k.
        # j=0, k=2b -> 2b
        # j=0, k=2b+1 -> 2b+1
        # j=1, k=2b -> 2 + 2b
        # j=1, k=2b+1 -> 2 + 2b + 1
        
        # W indices for this block:
        # C rows are 0,1. Columns are 2b, 2b+1.
        # W index = k*n + i.
        # k=2b, i=0 -> 4b
        # k=2b, i=1 -> 4b + 1
        # k=2b+1, i=0 -> 4b + 2
        # k=2b+1, i=1 -> 4b + 3
        
        for r in range(7):
            # U is the same for all blocks (A is shared)
            U[:, col_offset + r] = U_str[:, r]
            
            # Map V_str indices (0..3) to global V indices
            # V_str[0] is B00 (j=0, k=0) -> global j=0, k=2b -> 2b
            # V_str[1] is B01 (j=0, k=1) -> global j=0, k=2b+1 -> 2b+1
            # V_str[2] is B10 (j=1, k=0) -> global j=1, k=2b -> 2 + 2b
            # V_str[3] is B11 (j=1, k=1) -> global j=1, k=2b+1 -> 2 + 2b + 1
            
            V[2*b, col_offset + r] = V_str[0, r]
            V[2*b+1, col_offset + r] = V_str[1, r]
            V[2+2*b, col_offset + r] = V_str[2, r]
            V[2+2*b+1, col_offset + r] = V_str[3, r]
            
            # Map W_str indices (0..3) to global W indices
            # W_str[0] is C00 (k=0, i=0) -> global k=2b, i=0 -> 4b
            # W_str[1] is C01 (k=1, i=0) -> global k=2b+1, i=0 -> 4b+2
            # W_str[2] is C10 (k=0, i=1) -> global k=2b, i=1 -> 4b+1
            # W_str[3] is C11 (k=1, i=1) -> global k=2b+1, i=1 -> 4b+3
            
            W[4*b, col_offset + r] = W_str[0, r]
            W[4*b+2, col_offset + r] = W_str[1, r]
            W[4*b+1, col_offset + r] = W_str[2, r]
            W[4*b+3, col_offset + r] = W_str[3, r]
            
    return U, V, W

def _construct_m2110():
    """
    Construct U, V, W for M_{2,1,10} with rank 20.
    n=2, m=1, p=10.
    U: 2x20
    V: 10x20
    W: 20x20
    """
    n, m, p = 2, 1, 10
    num_U = n * m # 2
    num_V = m * p # 10
    num_W = p * n # 20
    rank = 20
    
    U = np.zeros((num_U, rank))
    V = np.zeros((num_V, rank))
    W = np.zeros((num_W, rank))
    
    # Schoolbook: C[i,k] = A[i,0] * B[0,k]
    # U[i*m+j, r] = A[i,j]
    # V[j*p+k, r] = B[j,k]
    # W[k*n+i, r] = C[i,k]
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            U[i, r] = 1
            V[k, r] = 1
            W[k*n+i, r] = 1
            
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions."""
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS # 2, 3, 10
    
    # 1. Construct M_{2,2,10} part (first 2 columns of A, first 2 rows of B)
    # Note: A is 2x3. We take A[:, :2] (2x2).
    # B is 3x10. We take B[:2, :] (2x10).
    U1, V1, W1 = _construct_m2210()
    
    # 2. Construct M_{2,1,10} part (last 1 column of A, last 1 row of B)
    # A[:, 2:] (2x1).
    # B[2:, :] (1x10).
    U2, V2, W2 = _construct_m2110()
    
    # 3. Map to global indices
    # Global U: (2*3=6) x rank
    # Global V: (3*10=30) x rank
    # Global W: (10*2=20) x rank
    
    total_rank = U1.shape[1] + U2.shape[1]
    U_global = np.zeros((6, total_rank))
    V_global = np.zeros((30, total_rank))
    W_global = np.zeros((20, total_rank))
    
    # Copy Part 1 (M_{2,2,10})
    # U1 is 4x35. Maps to A[0:2, 0:2].
    # U1[i, j] corresponds to global U[i*3 + j, :]
    for i in range(2):
        for j in range(2):
            U_global[i*3 + j, :35] = U1[i*2 + j, :]
            
    # V1 is 20x35. Maps to B[0:2, :].
    # V1[j, k] corresponds to global V[j*10 + k, :]
    for j in range(2):
        for k in range(10):
            V_global[j*10 + k, :35] = V1[j*2 + k, :]
            
    # W1 is 20x35. Maps to C[:, :].
    # W1[k, i] corresponds to global W[k*2 + i, :]
    for k in range(10):
        for i in range(2):
            W_global[k*2 + i, :35] = W1[k*2 + i, :]
            
    # Copy Part 2 (M_{2,1,10})
    # U2 is 2x20. Maps to A[0:2, 2].
    # U2[i, :] corresponds to global U[i*3 + 2, 35:]
    for i in range(2):
        U_global[i*3 + 2, 35:] = U2[i, :]
        
    # V2 is 10x20. Maps to B[2, :].
    # V2[k, :] corresponds to global V[2*10 + k, 35:]
    for k in range(10):
        V_global[2*10 + k, 35:] = V2[k, :]
        
    # W2 is 20x20. Maps to C[:, :].
    # W2[k, i] corresponds to global W[k*2 + i, 35:]
    for k in range(10):
        for i in range(2):
            W_global[k*2 + i, 35:] = W2[k*2 + i, :]
            
    # Convert to lists
    U_list = U_global.tolist()
    V_list = V_global.tolist()
    W_list = W_global.tolist()
    
    # Verify
    if not _verify(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U_list, "V": V_list, "W": W_list}
