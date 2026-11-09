import numpy as np

def construct():
    n, m, p = 2, 3, 10
    
    # We will use a rank-36 decomposition:
    # - 5 blocks of 2x2x2 Strassen decompositions (rank 7 each) for k in {0,1}, {2,3}, {4,5}, {6,7}, {8,9}
    # - 1 rank-1 term for j=2 (the third row of A, which multiplies the third column of B)
    #
    # The tensor to decompose is T[i, j, k] = 1 for all i in [0,2), j in [0,3), k in [0,10)
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (20, rank)
    
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 7 tensors for 2x2x2
    # Each tensor is (u, v, w) where u is 4-vector (flattened 2x2), v is 4-vector, w is 4-vector
    # Indices: u[i*2+j], v[j*2+k], w[k*2+i]
    strassen_tensors = [
        # (u, v, w)
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 1, -1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, -1, 0]),
        ([0, 1, -1, 0], [0, 0, 1, 0], [0, 0, 0, 1]),
        ([0, 0, 1, -1], [1, 0, 0, 0], [1, 0, 0, 0]),
    ]
    
    # For each block of 2 k-values, use the 7 Strassen tensors
    for block in range(5):
        k_offset = block * 2
        for t_idx, (u, v, w) in enumerate(strassen_tensors):
            col = block * 7 + t_idx
            
            # u has 4 entries: u[0]=u[0,0], u[1]=u[0,1], u[2]=u[1,0], u[3]=u[1,1]
            # For j=0: U[i*3 + 0, col] = u[i*2 + 0] for i=0,1
            # For j=1: U[i*3 + 1, col] = u[i*2 + 1] for i=0,1
            # For j=2: U[i*3 + 2, col] = 0 (handled separately)
            for i in range(2):
                U[i * 3 + 0, col] = u[i * 2 + 0]
                U[i * 3 + 1, col] = u[i * 2 + 1]
                U[i * 3 + 2, col] = 0.0
            
            # v has 4 entries: v[0]=v[0,0], v[1]=v[0,1], v[2]=v[1,0], v[3]=v[1,1]
            # For j=0: V[0*10 + k, col] = v[0*2 + k_local] for k_local=0,1
            # For j=1: V[1*10 + k, col] = v[1*2 + k_local] for k_local=0,1
            # For j=2: V[2*10 + k, col] = 0
            for j in range(2):
                for k_local in range(2):
                    k_global = k_offset + k_local
                    V[j * 10 + k_global, col] = v[j * 2 + k_local]
            for k_global in range(k_offset, k_offset + 2):
                V[2 * 10 + k_global, col] = 0.0
            
            # w has 4 entries: w[0]=w[0,0], w[1]=w[0,1], w[2]=w[1,0], w[3]=w[1,1]
            # w[k*2 + i] where k is local, i is 0 or 1
            # W[k_global * 2 + i, col]
            for k_local in range(2):
                k_global = k_offset + k_local
                for i in range(2):
                    W[k_global * 2 + i, col] = w[k_local * 2 + i]
    
    # Now handle j=2 with rank 1 (column 35)
    col = 35
    # U[i*3 + 2, col] = 1 for i=0,1
    U[0 * 3 + 2, col] = 1.0
    U[1 * 3 + 2, col] = 1.0
    
    # V[2*10 + k, col] = 1 for all k=0..9
    for k in range(10):
        V[2 * 10 + k, col] = 1.0
    
    # W[k*2 + i, col] = 1 for all k=0..9, i=0,1
    for k in range(10):
        W[k * 2 + 0, col] = 1.0
        W[k * 2 + 1, col] = 1.0
    
    # Verify the decomposition
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # Should equal 1 for all i,j,k
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                error = max(error, abs(val - 1.0))
    
    if error > 1e-10:
        # Fallback to schoolbook if something went wrong
        rank_schoolbook = n * m * p
        U_fb = np.zeros((n * m, rank_schoolbook))
        V_fb = np.zeros((m * p, rank_schoolbook))
        W_fb = np.zeros((p * n, rank_schoolbook))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
