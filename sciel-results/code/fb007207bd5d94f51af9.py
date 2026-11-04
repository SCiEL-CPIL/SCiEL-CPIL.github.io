import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target Rank 89 using Strassen for 2x2x2 block
    rank = 89
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen Block for i in {0,1}, j in {0,1}, k in {0,1}
    # Indices in U: i*m + j
    # Indices in V: j*p + k
    # Indices in W: k*n + i
    
    col = 0
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00 (0), A11 (1*m+1=6)
    # V: B00 (0), B11 (1*p+1=7)
    # W: C00 (0), C11 (1*n+1=4)
    U[0, col] = 1
    U[1*m+1, col] = 1
    V[0, col] = 1
    V[1*p+1, col] = 1
    W[0, col] = 1
    W[1*n+1, col] = 1
    col += 1
    
    # M2 = (A10 + A11)B00
    # U: A10 (1*m+0=5), A11 (6)
    # V: B00 (0)
    # W: C10 (0*n+1=1)
    U[1*m+0, col] = 1
    U[1*m+1, col] = 1
    V[0, col] = 1
    W[0*n+1, col] = 1
    col += 1
    
    # M3 = A00(B10 - B11)
    # U: A00 (0)
    # V: B10 (1*p+0=6), B11 (7)
    # W: C01 (1*n+0=3)
    U[0, col] = 1
    V[1*p+0, col] = 1
    V[1*p+1, col] = -1
    W[1*n+0, col] = 1
    col += 1
    
    # M4 = A11(B01 - B00)
    # U: A11 (6)
    # V: B01 (0*p+1=1), B00 (0)
    # W: C00 (0), C10 (1)
    U[1*m+1, col] = 1
    V[0*p+1, col] = 1
    V[0*p+0, col] = -1
    W[0, col] = 1
    W[0*n+1, col] = 1
    col += 1
    
    # M5 = (A00 + A01)B11
    # U: A00 (0), A01 (0*m+1=1)
    # V: B11 (7)
    # W: C00 (0) neg, C01 (3) pos
    U[0, col] = 1
    U[0*m+1, col] = 1
    V[1*p+1, col] = 1
    W[0, col] = -1
    W[1*n+0, col] = 1
    col += 1
    
    # M6 = (A01 - A00)(B00 + B10)
    # U: A01 (1) pos, A00 (0) neg
    # V: B00 (0) pos, B10 (6) pos
    # W: C11 (4) neg
    U[0*m+1, col] = 1
    U[0, col] = -1
    V[0, col] = 1
    V[1*p+0, col] = 1
    W[1*n+1, col] = -1
    col += 1
    
    # M7 = (A10 - A11)(B11 + B01)
    # U: A10 (5) pos, A11 (6) neg
    # V: B11 (7) pos, B01 (1) pos
    # W: C00 (0) neg, C11 (4) pos
    U[1*m+0, col] = 1
    U[1*m+1, col] = -1
    V[1*p+1, col] = 1
    V[0*p+1, col] = 1
    W[0, col] = -1
    W[1*n+1, col] = 1
    col += 1
    
    # Remaining Schoolbook terms for all (i,j,k) NOT in {0,1}^3
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in [0,1] and j in [0,1] and k in [0,1]:
                    continue
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    # Verify
    # Pick a random matrix
    np.random.seed(42)
    A = np.random.randint(-5, 5, (n, m))
    B = np.random.randint(-5, 5, (m, p))
    C_ref = A @ B
    
    # Compute C_decomp
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        # u is vector of length n*m
        # v is vector of length m*p
        # w is vector of length p*n
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        # Reconstruct contribution
        # The term is sum_{j} U[i, j] V[j, k] * W[k, i]?
        # No, the bilinear form is:
        # For each i, k:
        #   sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_decomp[i, k] += val
                
    if not np.allclose(C_ref, C_decomp):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
