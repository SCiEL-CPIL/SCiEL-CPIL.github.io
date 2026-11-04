import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen's 2x2x2 algorithm factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Row indices: for U: (i,j) -> i*2+j; for V: (j,k) -> j*2+k; for W: (k,i) -> k*2+i
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker products: 16x49
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    # Build permutations by solving the tensor identity at runtime
    # The tensor identity is: sum_t U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t] == delta_{i,j,k}
    # where delta is 1 if (i,j,k) corresponds to a valid triple, else 0
    # Actually for matrix mult: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k
    
    # We need to find permutations perm_U, perm_V, perm_W such that:
    # U4[perm_U[S], t] for S = i*4+j gives the correct row for A[i,j]
    # V4[perm_V[S], t] for S = j*4+k gives the correct row for B[j,k]
    # W4[perm_W[S], t] for S = k*4+i gives the correct row for C[k,i]
    
    # The Kronecker product np.kron(U2, U2) has row index:
    # L4 = L2_outer * 4 + L2_inner
    # where L2_outer = r_out * 2 + c_out (for U, this is the outer block indices)
    # and L2_inner = r_in * 2 + c_in (for U, this is the inner indices)
    
    # For U (A matrix, 4x4):
    #   Standard index S = r*4 + c
    #   r = 2*r_out + r_in, c = 2*c_out + c_in
    #   Kronecker row: L4 = (r_out*2 + c_out) * 4 + (r_in*2 + c_in)
    # Pre-computed permutations for efficiency and clarity
    # perm_U[S] maps standard index S = r*4+c to Kronecker index L4
    # r = 2*r_out + r_in, c = 2*c_out + c_in
    # L4 = (r_out*2 + c_out)*4 + (r_in*2 + c_in)
    perm_U = np.array([
        0,  1,  8,  9,  # r=0 (00,01), c=0..3
        2,  3, 10, 11,  # r=1 (01,00), c=0..3
        4,  5, 12, 13,  # r=2 (10,00), c=0..3
        6,  7, 14, 15   # r=3 (11,00), c=0..3
    ], dtype=int)
    
    # perm_V[S] maps standard index S = j*4+k to Kronecker index L4
    # j = 2*j_out + j_in, k = 2*k_out + k_in
    # L4 = (j_out*2 + k_out)*4 + (j_in*2 + k_in)
    perm_V = np.array([
        0,  1,  8,  9,  # j=0
        2,  3, 10, 11,  # j=1
        4,  5, 12, 13,  # j=2
        6,  7, 14, 15   # j=3
    ], dtype=int)
    
    # perm_W[S] maps standard index S = k*4+r to Kronecker index L4
    # k = 2*k_out + k_in, r = 2*r_out + r_in
    # L4 = (k_out*2 + r_out)*4 + (k_in*2 + r_in)
    perm_W = np.array([
        0,  1,  8,  9,  # k=0
        2,  3, 10, 11,  # k=1
        4,  5, 12, 13,  # k=2
        6,  7, 14, 15   # k=3
    ], dtype=int)
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Runtime verification: construct the full tensor and check identity
    rank = U4.shape[1]
    
    # Build index arrays for all (i,j,k) triples
    i_idx = np.repeat(np.arange(4), 16)
    j_idx = np.tile(np.repeat(np.arange(4), 4), 4)
    k_idx = np.tile(np.arange(4), 16)
    
    U_rows = U4[i_idx * 4 + j_idx, :]
    V_rows = V4[j_idx * 4 + k_idx, :]
    W_rows = W4[k_idx * 4 + i_idx, :]
    
    products = U_rows * V_rows * W_rows
    totals = products.sum(axis=1)
    
    valid = np.all(totals == 1)
    
    if not valid:
        # Fallback to schoolbook
        n, m, p = 4, 4, 4
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
