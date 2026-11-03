import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We split dimensions to exploit Strassen and low-rank properties of 1-dim blocks.
    # n=3 -> 1 + 2
    # m=5 -> 2 + 3
    # p=6 -> 2 + 2 + 2
    
    n_splits = [1, 2]
    m_splits = [2, 3]
    p_splits = [2, 2, 2]
    
    # Accumulate U, V, W columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a block decomposition
    def add_block(i_off, j_off, k_off, n_i, m_j, p_k, block_type):
        """
        Adds decomposition for block A[i_off:i_off+n_i, j_off:j_off+m_j] * B[j_off:j_off+m_j, k_off:k_off+p_k]
        to the global U, V, W.
        """
        # Local indices
        u_rows = n_i * m_j
        v_rows = m_j * p_k
        w_rows = p_k * n_i
        
        U_block = np.zeros((u_rows, 0))
        V_block = np.zeros((v_rows, 0))
        W_block = np.zeros((w_rows, 0))
        
        if block_type == 'strassen':
            # Must be 2x2x2
            # Strassen decomposition
            # M1 = A11(A22-A12)
            # M2 = (A11+A12)B22
            # M3 = (A21+A22)B11
            # M4 = A22(A12-A21)
            # M5 = (A11+A22)(B11+B22)
            # M6 = (A11-A21)(B12+B21)
            # M7 = (A22-A12)(B11-B22)
            # C11 = M5+M4-M3+M1
            # C12 = M2+M6
            # C21 = M3+M7
            # C22 = M5+M1-M4-M6
            
            # U vectors (coeffs for A)
            # V vectors (coeffs for B)
            # W vectors (coeffs for C)
            
            # Indices:
            # A: 11, 12, 21, 22 -> 0, 1, 2, 3
            # B: 11, 12, 21, 22 -> 0, 1, 2, 3
            # C: 11, 12, 21, 22 -> 0, 1, 2, 3
            
            terms = [
                # (A_coeffs, B_coeffs, C_coeffs)
                # M1: A11(A22-A12) -> A=[1,0,0,0], B=[0,-1,0,1] ? No, B is (B22-B12).
                # B indices: 11, 12, 21, 22.
                # B22-B12 -> [0, -1, 0, 1]
                ([1, 0, 0, 0], [0, -1, 0, 1], [1, 0, 0, 0]), # C11 += M1
                # M2: (A11+A12)B22 -> A=[1,1,0,0], B=[0,0,0,1]
                ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]), # C12 += M2
                # M3: (A21+A22)B11 -> A=[0,0,1,1], B=[1,0,0,0]
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # C21 += M3
                # M4: A22(A12-A21) -> A=[0,0,0,1], B=[0,1,-1,0]
                ([0, 0, 0, 1], [0, 1, -1, 0], [-1, 0, 0, 1]), # C11 -= M4, C22 += M4 ?
                # Wait, C11 = M5+M4-M3+M1. C22 = M5+M1-M4-M6.
                # So M4 adds to C11 and subtracts from C22.
                ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, -1]),
                # M5: (A11+A22)(B11+B22) -> A=[1,0,0,1], B=[1,0,0,1]
                ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C11+=M5, C22+=M5
                # M6: (A11-A21)(B12+B21) -> A=[1,0,-1,0], B=[0,1,1,0]
                ([1, 0, -1, 0], [0, 1, 1, 0], [0, 1, 0, -1]), # C12+=M6, C22-=M6
                # M7: (A22-A12)(B11-B22) -> A=[0,-1,0,1], B=[1,0,0,-1]
                ([0, -1, 0, 1], [1, 0, 0, -1], [0, 0, 1, 0]), # C21+=M7
            ]
            
            for a_coeff, b_coeff, c_coeff in terms:
                col_U = np.zeros(n_i * m_j)
                col_V = np.zeros(m_j * p_k)
                col_W = np.zeros(p_k * n_i)
                
                for idx_a, val_a in enumerate(a_coeff):
                    # a_coeff index corresponds to A local indices
                    # 0:11, 1:12, 2:21, 3:22
                    r, c = divmod(idx_a, m_j) # r is row, c is col
                    col_U[r * m_j + c] = val_a
                    
                for idx_b, val_b in enumerate(b_coeff):
                    # b_coeff index corresponds to B local indices
                    # 0:11, 1:12, 2:21, 3:22
                    r, c = divmod(idx_b, p_k)
                    col_V[r * p_k + c] = val_b
                    
                for idx_c, val_c in enumerate(c_coeff):
                    # c_coeff index corresponds to C local indices
                    # 0:11, 1:12, 2:21, 3:22
                    # W index is k*n + i
                    r, c = divmod(idx_c, n_i) # r is k (col of B/row of C), c is i (row of A/col of C)
                    # In our W definition: index k*n + i.
                    # Here r is row of C (which is k), c is col of C (which is i).
                    col_W[r * n_i + c] = val_c
                
                U_block = np.column_stack([U_block, col_U])
                V_block = np.column_stack([V_block, col_V])
                W_block = np.column_stack([W_block, col_W])
                
        elif block_type == 'linear_1_m_p':
            # n_i = 1
            # Rank = m_j
            # Terms: for each j, A[0,j] * B[j,k]
            # U[0,j] = 1
            # V[j,k] = 1
            # W[k,0] = 1
            rank = m_j
            for j in range(m_j):
                col_U = np.zeros(n_i * m_j)
                col_V = np.zeros(m_j * p_k)
                col_W = np.zeros(p_k * n_i)
                
                col_U[0 * m_j + j] = 1 # i=0, j=j
                
                for k in range(p_k):
                    col_V[j * p_k + k] = 1 # j=j, k=k
                    col_W[k * n_i + 0] = 1 # k=k, i=0
                
                U_block = np.column_stack([U_block, col_U])
                V_block = np.column_stack([V_block, col_V])
                W_block = np.column_stack([W_block, col_W])
                
        elif block_type == 'linear_m_1_p':
            # m_j = 1
            # Rank = n_i? No, Rank = p_k?
            # C[i,k] = A[i,0] B[0,k]
            # Terms: for each i, A[i,0] * B[0,k]?
            # U[i,0] = 1
            # V[0,k] = 1
            # W[k,i] = 1
            # This is rank n_i?
            # If n_i=2, p_k=2.
            # T = A1 B1 + A2 B2 ? No.
            # C11 = A11 B11. C12 = A11 B12. C21 = A21 B11. C22 = A21 B12.
            # Rank 2.
            # t=0: A11 term. U[1,0]=1, V[0,:]=1, W[:,1]=1.
            # t=1: A21 term. U[2,0]=1, V[0,:]=1, W[:,2]=1.
            rank = n_i
            for i in range(n_i):
                col_U = np.zeros(n_i * m_j)
                col_V = np.zeros(m_j * p_k)
                col_W = np.zeros(p_k * n_i)
                
                col_U[i * m_j + 0] = 1 # i=i, j=0
                
                for k in range(p_k):
                    col_V[0 * p_k + k] = 1 # j=0, k=k
                    col_W[k * n_i + i] = 1 # k=k, i=i
                    
                U_block = np.column_stack([U_block, col_U])
                V_block = np.column_stack([V_block, col_V])
                W_block = np.column_stack([W_block, col_W])
                
        else:
            # Schoolbook
            rank = n_i * m_j * p_k
            for i in range(n_i):
                for j in range(m_j):
                    for k in range(p_k):
                        col_U = np.zeros(n_i * m_j)
                        col_V = np.zeros(m_j * p_k)
                        col_W = np.zeros(p_k * n_i)
                        
                        col_U[i * m_j + j] = 1
                        col_V[j * p_k + k] = 1
                        col_W[k * n_i + i] = 1
                        
                        U_block = np.column_stack([U_block, col_U])
                        V_block = np.column_stack([V_block, col_V])
                        W_block = np.column_stack([W_block, col_W])
                        
        if U_block.shape[1] > 0:
            U_cols.append((i_off * m + j_off, U_block)) # Start row in global U
            V_cols.append((j_off * p + k_off, V_block)) # Start row in global V
            W_cols.append((k_off * n + i_off, W_block)) # Start row in global W

    # Iterate through blocks
    i_cum = 0
    for n_i in n_splits:
        j_cum = 0
        for m_j in m_splits:
            k_cum = 0
            for p_k in p_splits:
                # Determine block type
                if n_i == 2 and m_j == 2 and p_k == 2:
                    type = 'strassen'
                elif n_i == 1:
                    type = 'linear_1_m_p'
                elif m_j == 1:
                    type = 'linear_m_1_p'
                else:
                    type = 'schoolbook'
                
                add_block(i_cum, j_cum, k_cum, n_i, m_j, p_k, type)
                k_cum += p_k
            j_cum += m_j
        i_cum += n_i
        
    # Construct final matrices
    total_rank = sum(len(u[1]) for u in U_cols)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for (start_u, u_block), (start_v, v_block), (start_w, w_block) in zip(U_cols, V_cols, W_cols):
        num_cols = u_block.shape[1]
        U[start_u:start_u+u_block.shape[0], col_offset:col_offset+num_cols] = u_block
        V[start_v:start_v+v_block.shape[0], col_offset:col_offset+num_cols] = v_block
        W[start_w:start_w+w_block.shape[0], col_offset:col_offset+num_cols] = w_block
        col_offset += num_cols
        
    # Convert to lists for interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
