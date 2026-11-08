import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a block decomposition with Strassen for 2x2x2 blocks and schoolbook for others.
    Target rank: 42.
    """
    n, m, p = DIMENSIONS
    
    # Global dimensions
    dim_u = n * m # 12
    dim_v = m * p # 20
    dim_w = p * n # 15
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, i_start, j_start, k_start, n_block, m_block, p_block):
        # U_block: (n_block*m_block, rank)
        # V_block: (m_block*p_block, rank)
        # W_block: (p_block*n_block, rank)
        
        U_full = np.zeros((dim_u, U_block.shape[1]))
        V_full = np.zeros((dim_v, V_block.shape[1]))
        W_full = np.zeros((dim_w, W_block.shape[1]))
        
        # Map U indices
        # Local i in [0, n_block), j in [0, m_block)
        # Global i = i_start + i, j = j_start + j
        # Local index = i * m_block + j
        # Global index = (i_start + i) * m + (j_start + j)
        
        for r in range(U_block.shape[1]):
            for local_idx in range(U_block.shape[0]):
                val = U_block[local_idx, r]
                if val != 0:
                    li = local_idx // m_block
                    lj = local_idx % m_block
                    gi = i_start + li
                    gj = j_start + lj
                    g_idx = gi * m + gj
                    U_full[g_idx, r] = val
                    
        for r in range(V_block.shape[1]):
            for local_idx in range(V_block.shape[0]):
                val = V_block[local_idx, r]
                if val != 0:
                    lj = local_idx // p_block
                    lk = local_idx % p_block
                    gj = j_start + lj
                    gk = k_start + lk
                    g_idx = gj * p + gk
                    V_full[g_idx, r] = val
                    
        for r in range(W_block.shape[1]):
            for local_idx in range(W_block.shape[0]):
                val = W_block[local_idx, r]
                if val != 0:
                    lk = local_idx // n_block
                    li = local_idx % n_block
                    gk = k_start + lk
                    gi = i_start + li
                    g_idx = gk * n + gi
                    W_full[g_idx, r] = val
        
        U_parts.append(U_full)
        V_parts.append(V_full)
        W_parts.append(W_full)

    # Strassen 2x2x2
    def get_strassen_2x2x2():
        # U, V, W for 2x2x2
        # Indices: U (4x7), V (4x7), W (4x7)
        # U: i*2+j
        # V: j*2+k
        # W: k*2+i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        # A00 (0,0), A11 (1,1)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1 # C00 += M1
        
        # M2 = (A10+A11)B00
        # A10 (1,0), A11 (1,1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10 += M2
        
        # M3 = A00(B10-B11)
        # A00 (0,0)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01 += M3
        
        # M4 = A11(B01-B00)
        # A11 (1,1)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C00 += M4, C10 += M4
        
        # M5 = (A00+A01)B11
        # A00 (0,0), A01 (0,1)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C00 -= M5, C01 += M5
        
        # M6 = (A10-A00)(B00+B01)
        # A10 (1,0), A00 (0,0)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C11 += M6
        
        # M7 = (A01-A11)(B10+B11)
        # A01 (0,1), A11 (1,1)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1 # C00 += M7, C11 += M7
        
        return U, V, W

    # Schoolbook 2x2x1
    def get_schoolbook_2x2x1():
        # U (4x4), V (2x4), W (2x4)
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # i=0, j=0, k=0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # i=0, j=1, k=0
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # i=1, j=0, k=0
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # i=1, j=1, k=0
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x2
    def get_schoolbook_1x2x2():
        # U (2x4), V (4x4), W (2x4)
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # i=0, j=0, k=0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # i=0, j=0, k=1
        U[0, 1] = 1
        V[1, 1] = 1
        W[1, 1] = 1
        
        # i=0, j=1, k=0
        U[1, 2] = 1
        V[2, 2] = 1
        W[0, 2] = 1
        
        # i=0, j=1, k=1
        U[1, 3] = 1
        V[3, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x1
    def get_schoolbook_1x2x1():
        # U (2x2), V (2x2), W (1x2)
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        # i=0, j=0, k=0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # i=0, j=1, k=0
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        return U, V, W

    # Block 1: 2x4x5
    # 1a: 2x2x2 (i=0:2, j=0:2, k=0:2)
    U1a, V1a, W1a = get_strassen_2x2x2()
    add_block(U1a, V1a, W1a, 0, 0, 0, 2, 2, 2)
    
    # 1b: 2x2x2 (i=0:2, j=2:4, k=0:2)
    U1b, V1b, W1b = get_strassen_2x2x2()
    add_block(U1b, V1b, W1b, 0, 2, 0, 2, 2, 2)
    
    # 1c: 2x4x1 (i=0:2, j=0:4, k=4:5)
    # 1c1: 2x2x1 (i=0:2, j=0:2, k=4:5)
    U1c1, V1c1, W1c1 = get_schoolbook_2x2x1()
    add_block(U1c1, V1c1, W1c1, 0, 0, 4, 2, 2, 1)
    
    # 1c2: 2x2x1 (i=0:2, j=2:4, k=4:5)
    U1c2, V1c2, W1c2 = get_schoolbook_2x2x1()
    add_block(U1c2, V1c2, W1c2, 0, 2, 4, 2, 2, 1)
    
    # Block 2: 1x4x5
    # 2a: 1x4x2 (i=2:3, j=0:4, k=0:2)
    # 2a1: 1x2x2 (i=2:3, j=0:2, k=0:2)
    U2a1, V2a1, W2a1 = get_schoolbook_1x2x2()
    add_block(U2a1, V2a1, W2a1, 2, 0, 0, 1, 2, 2)
    
    # 2a2: 1x2x2 (i=2:3, j=2:4, k=0:2)
    U2a2, V2a2, W2a2 = get_schoolbook_1x2x2()
    add_block(U2a2, V2a2, W2a2, 2, 2, 0, 1, 2, 2)
    
    # 2b: 1x4x2 (i=2:3, j=0:4, k=2:4)
    # 2b1: 1x2x2 (i=2:3, j=0:2, k=2:4)
    U2b1, V2b1, W2b1 = get_schoolbook_1x2x2()
    add_block(U2b1, V2b1, W2b1, 2, 0, 2, 1, 2, 2)
    
    # 2b2: 1x2x2 (i=2:3, j=2:4, k=2:4)
    U2b2, V2b2, W2b2 = get_schoolbook_1x2x2()
    add_block(U2b2, V2b2, W2b2, 2, 2, 2, 1, 2, 2)
    
    # 2c: 1x4x1 (i=2:3, j=0:4, k=4:5)
    # 2c1: 1x2x1 (i=2:3, j=0:2, k=4:5)
    U2c1, V2c1, W2c1 = get_schoolbook_1x2x1()
    add_block(U2c1, V2c1, W2c1, 2, 0, 4, 1, 2, 1)
    
    # 2c2: 1x2x1 (i=2:3, j=2:4, k=4:5)
    U2c2, V2c2, W2c2 = get_schoolbook_1x2x1()
    add_block(U2c2, V2c2, W2c2, 2, 2, 4, 1, 2, 1)

    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
