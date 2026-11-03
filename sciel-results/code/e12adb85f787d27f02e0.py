import numpy as np

DIMENSIONS = [2, 4, 7]

def _build_strassen_222():
    """
    Return (U, V, W) for 2x2 by 2x2 matrix multiplication using Strassen.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
      A[i, j] -> U[i*2 + j]
      B[j, k] -> V[j*2 + k]
      C[i, k] -> W[k*2 + i]  (trace convention W[k*n+i] with n=2)
    """
    n, m, p = 2, 2, 2
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Column 0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11 (k=0, i=0 -> W[0*2+0]=W[0])
    W[3, 0] = 1  # C22 (k=1, i=1 -> W[1*2+1]=W[3])
    
    # Column 1: M2 = (A21+A22)B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21 (k=0, i=1 -> W[0*2+1]=W[2])
    W[3, 1] = -1 # C22 (k=1, i=1 -> W[3])
    
    # Column 2: M3 = A11(B21-B22)
    U[0, 2] = 1  # A11
    V[2, 2] = 1  # B21
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12 (k=1, i=0 -> W[1*2+0]=W[2])... wait
    # Let's recheck W indexing: W[k*n+i]
    # C11: i=0,k=0 -> W[0]
    # C12: i=0,k=1 -> W[2]
    # C21: i=1,k=0 -> W[1]
    # C22: i=1,k=1 -> W[3]
    
    # Correcting W indices:
    # C11 -> W[0]
    # C12 -> W[2]
    # C21 -> W[1]
    # C22 -> W[3]
    
    # Re-initialize and fill correctly
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)(B11+B22)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1  # C11
    W[3, 0] += 1  # C22
    
    # M2: (A21+A22)B11
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1  # C21
    W[3, 1] += -1 # C22
    
    # M3: A11(B21-B22)
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] += -1
    W[2, 2] += 1  # C12
    
    # M4: A22(B12-B11)
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] += -1
    W[0, 3] += 1  # C11
    W[1, 3] += 1  # C21
    
    # M5: (A11+A12)B22
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += -1 # C11
    W[2, 4] += 1  # C12
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1  # C22
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1  # C11
    
    return U, V, W

def _build_dot_2x2x2():
    """
    Rank-4 decomposition for 2x2 by 2x2 (schoolbook).
    """
    n, m, p = 2, 2, 2
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W

def construct():
    n, m, p = 2, 4, 7
    total_rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a block decomposition
    def add_block(local_U, local_V, local_W, i_offset, j_offset, k_offset):
        nonlocal total_rank
        r = local_U.shape[1]
        U_g = np.zeros((n*m, r))
        V_g = np.zeros((m*p, r))
        W_g = np.zeros((p*n, r))
        
        # Map local indices to global
        # local_U[i', j'] corresponds to global A[i_offset+i', j_offset+j']
        # local_V[j', k'] corresponds to global B[j_offset+j', k_offset+k']
        # local_W[k', i'] corresponds to global C[i_offset+i', k_offset+k']
        
        # local_U is 2x2 (i' in 0..1, j' in 0..1)
        # local_V is 2x2 (j' in 0..1, k' in 0..1)
        # local_W is 2x2 (k' in 0..1, i' in 0..1)
        
        for i_l in range(2):
            for j_l in range(2):
                idx_u = i_l * 2 + j_l
                g_i = i_offset + i_l
                g_j = j_offset + j_l
                U_g[g_i * m + g_j, :] = local_U[idx_u, :]
                
        for j_l in range(2):
            for k_l in range(2):
                idx_v = j_l * 2 + k_l
                g_j = j_offset + j_l
                g_k = k_offset + k_l
                V_g[g_j * p + g_k, :] = local_V[idx_v, :]
                
        for k_l in range(2):
            for i_l in range(2):
                idx_w = k_l * 2 + i_l
                g_k = k_offset + k_l
                g_i = i_offset + i_l
                W_g[g_k * n + g_i, :] = local_W[idx_w, :]
                
        U_parts.append(U_g)
        V_parts.append(V_g)
        W_parts.append(W_g)
        total_rank += r

    # Block 1: A[:, 0:2] * B[0:2, :] (2x2 * 2x7)
    # Split B[0:2, :] into B[0:2, 0:2] (2x2), B[0:2, 2:4] (2x2), B[0:2, 4:6] (2x2), B[0:2, 6:7] (2x1)
    
    # Sub-block 1.1: A[:, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> Strassen (rank 7)
    U1, V1, W1 = _build_strassen_222()
    add_block(U1, V1, W1, i_offset=0, j_offset=0, k_offset=0)
    
    # Sub-block 1.2: A[:, 0:2] (2x2) * B[0:2, 2:4] (2x2) -> Strassen (rank 7)
    U2, V2, W2 = _build_strassen_222()
    add_block(U2, V2, W2, i_offset=0, j_offset=0, k_offset=2)
    
    # Sub-block 1.3: A[:, 0:2] (2x2) * B[0:2, 4:6] (2x2) -> Strassen (rank 7)
    U3, V3, W3 = _build_strassen_222()
    add_block(U3, V3, W3, i_offset=0, j_offset=0, k_offset=4)
    
    # Sub-block 1.4: A[:, 0:2] (2x2) * B[0:2, 6:7] (2x1) -> Rank 4 (schoolbook 2x2x1)
    # This is 4 multiplications
    U4 = np.zeros((4, 4))
    V4 = np.zeros((2, 4))  # V is 2x1, so 2 rows
    W4 = np.zeros((2, 4))  # W is 2x2 (k=0..1, i=0..1) -> 4 entries, but k is only 6? No, k_offset=6, so k is just 6.
    # Wait, the block is 2x1. The output is 2x1.
    # Indices for V: j in 0..1, k in 0..0 (local). Global k=6.
    # Indices for W: k in 0..0 (local), i in 0..1. Global k=6.
    # So W should be shape (1*2, 4) = (2, 4).
    # My add_block assumes 2x2 blocks. I need a separate handler or modify add_block.
    
    # Let's handle the 2x1 case manually to avoid confusion.
    # A is 2x2 (rows 0,1; cols 0,1)
    # B is 2x1 (rows 0,1; col 0)
    # C is 2x1 (rows 0,1; col 0)
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    
    U4 = np.zeros((4, 4))
    V4 = np.zeros((2, 4)) # 2 rows (j=0,1), 1 col (k=0) -> 2 entries per rank col
    W4 = np.zeros((2, 4)) # 1 col (k=0), 2 rows (i=0,1) -> 2 entries per rank col
    
    for i in range(2):
        for j in range(2):
            c = i * 2 + j
            U4[i*2+j, c] = 1
            V4[j, c] = 1
            W4[0*2+i, c] = 1 # k_local=0, so global k=6. W index is k*n+i = 6*2+i = 12+i?
            # Wait, W4 is local. The add_block maps it.
            # Local W index: k_local * n_local + i_local. Here n_local=2.
            # k_local=0, i_local=i. So index is i.
            # My W4 shape is (2, 4). Indices 0, 1. Correct.
            
    # Add to global
    r4 = 4
    U_g = np.zeros((n*m, r4))
    V_g = np.zeros((m*p, r4))
    W_g = np.zeros((p*n, r4))
    
    for i_l in range(2):
        for j_l in range(2):
            idx_u = i_l * 2 + j_l
            g_i = 0 + i_l
            g_j = 0 + j_l
            U_g[g_i * m + g_j, :] = U4[idx_u, :]
            
    for j_l in range(2):
        # k_l is only 0
        idx_v = j_l * 1 + 0 # V4 has 2 rows, 1 col. Index is just j_l.
        # Wait, V4 is (2, 4). Index is row j_l.
        g_j = 0 + j_l
        g_k = 6 + 0
        V_g[g_j * p + g_k, :] = V4[j_l, :]
        
    for i_l in range(2):
        # k_l is only 0
        idx_w = 0 * 2 + i_l # W4 is (2, 4). Index is i_l.
        g_k = 6 + 0
        g_i = 0 + i_l
        W_g[g_k * n + g_i, :] = W4[idx_w, :]
        
    U_parts.append(U_g)
    V_parts.append(V_g)
    W_parts.append(W_g)
    total_rank += r4
    
    # Block 2: A[:, 2:4] (2x2) * B[2:4, :] (2x7)
    # Similar to Block 1, but j_offset=2
    
    # Sub-block 2.1: A[:, 2:4] * B[2:4, 0:2] -> Strassen
    U5, V5, W5 = _build_strassen_222()
    add_block(U5, V5, W5, i_offset=0, j_offset=2, k_offset=0)
    
    # Sub-block 2.2: A[:, 2:4] * B[2:4, 2:4] -> Strassen
    U6, V6, W6 = _build_strassen_222()
    add_block(U6, V6, W6, i_offset=0, j_offset=2, k_offset=2)
    
    # Sub-block 2.3: A[:, 2:4] * B[2:4, 4:6] -> Strassen
    U7, V7, W7 = _build_strassen_222()
    add_block(U7, V7, W7, i_offset=0, j_offset=2, k_offset=4)
    
    # Sub-block 2.4: A[:, 2:4] * B[2:4, 6:7] -> Rank 4
    U8 = np.zeros((4, 4))
    V8 = np.zeros((2, 4))
    W8 = np.zeros((2, 4))
    for i in range(2):
        for j in range(2):
            c = i * 2 + j
            U8[i*2+j, c] = 1
            V8[j, c] = 1
            W8[i, c] = 1
            
    U_g = np.zeros((n*m, 4))
    V_g = np.zeros((m*p, 4))
    W_g = np.zeros((p*n, 4))
    for i_l in range(2):
        for j_l in range(2):
            idx_u = i_l * 2 + j_l
            g_i = 0 + i_l
            g_j = 2 + j_l
            U_g[g_i * m + g_j, :] = U8[idx_u, :]
    for j_l in range(2):
        g_j = 2 + j_l
        g_k = 6 + 0
        V_g[g_j * p + g_k, :] = V8[j_l, :]
    for i_l in range(2):
        g_k = 6 + 0
        g_i = 0 + i_l
        W_g[g_k * n + g_i, :] = W8[i_l, :]
        
    U_parts.append(U_g)
    V_parts.append(V_g)
    W_parts.append(W_g)
    total_rank += 4

    # Concatenate
    U_final = np.hstack(U_parts)
    V_final = np.hstack(V_parts)
    W_final = np.hstack(W_parts)
    
    # Convert to lists of lists with integer values (rounding tiny floats)
    U_list = U_final.astype(int).tolist()
    V_list = V_final.astype(int).tolist()
    W_list = W_final.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
