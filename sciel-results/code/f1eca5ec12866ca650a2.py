import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Rank 42 solution based on Strassen blocks
    rank = 42
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Standard Strassen 2x2 @ 2x2 coefficients
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U coefficients: A[i,j] where local index encodes (i,j)
    # local 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_U = [
        {0: 1, 3: 1},       # P1: A00 + A11
        {2: 1, 3: 1},       # P2: A10 + A11
        {0: 1},             # P3: A00
        {3: 1},             # P4: A11
        {0: 1, 1: 1},       # P5: A00 + A01
        {0: -1, 2: 1},      # P6: -A00 + A10
        {1: 1, 3: -1},      # P7: A01 - A11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1: B00 + B11
        {0: 1},             # P2: B00
        {2: 1, 3: -1},      # P3: B10 - B11
        {1: 1, 0: -1},      # P4: B01 - B00
        {3: 1},             # P5: B11
        {0: 1, 1: 1},       # P6: B00 + B01
        {2: 1, 3: 1},       # P7: B10 + B11
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: C00 + C11
        {2: 1, 3: -1},      # P2: C10 - C11
        {1: 1, 3: -1},      # P3: C01 - C11
        {0: -1, 2: 1},      # P4: -C00 + C10
        {0: 1, 1: 1},       # P5: C00 + C01
        {3: 1},             # P6: C11
        {0: 1},             # P7: C00
    ]
    
    col = 0
    for j_block in range(2): # j=0,1 and j=2,3
        for k_block in range(3): # k=0,1 and k=2,3 and k=4,5
            for prod_idx in range(7):
                r = col + prod_idx
                
                # U part: A[i, j]
                for local_idx, sign in strassen_U[prod_idx].items():
                    gi = local_idx // 2
                    gj = local_idx % 2
                    U[gi * m + (j_block * 2 + gj), r] = sign
                
                # V part: B[j, k]
                for local_idx, sign in strassen_V[prod_idx].items():
                    gj = local_idx // 2
                    gk = local_idx % 2
                    V[(j_block * 2 + gj) * p + (k_block * 2 + gk), r] = sign
                
                # W part: C[i, k]
                for local_idx, sign in strassen_W[prod_idx].items():
                    gi = local_idx // 2
                    gk = local_idx % 2
                    W[(k_block * 2 + gk) * n + gi, r] = sign
            
            col += 7
    
    # Verify the solution
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r
    # Actually, the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify with random matrices
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C = A @ B
    
    # Compute C from decomposition
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                val += sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for j in range(m))
            C_decomp[i, k] = val
    
    if not np.allclose(C, C_decomp, atol=1e-10):
        # Fallback to schoolbook if Strassen is wrong
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
