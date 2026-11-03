import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # Define block splits
    # A: 4x5 -> rows [0:2], [2:4]; cols [0:2], [2:5]
    # B: 5x5 -> rows [0:2], [2:5]; cols [0:2], [2:5]
    
    # We will accumulate the factors
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U_factors = []
    V_factors = []
    W_factors = []
    
    def add_block(n_blk, m_blk, p_blk, u_offset, v_offset, w_offset, gen_func):
        """
        Adds factors for a block multiplication.
        n_blk, m_blk, p_blk: dimensions of the block
        u_offset: (i_offset, j_offset) in global A
        v_offset: (j_offset, k_offset) in global B
        w_offset: (k_offset, i_offset) in global C
        gen_func: function returning U, V, W for the local block
        """
        U_local, V_local, W_local = gen_func(n_blk, m_blk, p_blk)
        
        # Map local indices to global
        # Local U index: i_l * m_blk + j_l
        # Global U index: (i_l + u_offset[0]) * m + (j_l + u_offset[1])
        
        # We need to expand the local factors into global size
        U_g = np.zeros((n * m, n_blk * m_blk))
        V_g = np.zeros((m * p, m_blk * p_blk))
        W_g = np.zeros((p * n, p_blk * n_blk))
        
        # This approach is inefficient if we create full arrays.
        # Better to just append to lists of non-zero contributions or build sparse-like structures.
        # Given small size, we can just build the full arrays and add.
        
        # Actually, let's just build the full U, V, W arrays for the whole problem
        # and add the block contributions to them.
        
        for r in range(U_local.shape[1]):
            for i_l in range(n_blk):
                for j_l in range(m_blk):
                    val = U_local[i_l * m_blk + j_l, r]
                    if val != 0:
                        i_g = i_l + u_offset[0]
                        j_g = j_l + u_offset[1]
                        U_g[(i_g) * m + j_g, r] = val
                        
            for j_l in range(m_blk):
                for k_l in range(p_blk):
                    val = V_local[j_l * p_blk + k_l, r]
                    if val != 0:
                        j_g = j_l + v_offset[0]
                        k_g = k_l + v_offset[1]
                        V_g[(j_g) * p + k_g, r] = val
                        
            for k_l in range(p_blk):
                for i_l in range(n_blk):
                    val = W_local[k_l * n_blk + i_l, r]
                    if val != 0:
                        k_g = k_l + w_offset[1] # w_offset is (k_offset, i_offset)
                        i_g = i_l + w_offset[0]
                        W_g[(k_g) * n + i_g, r] = val
                        
        return U_g, V_g, W_g

    # This add_block function is getting complex.
    # Simpler: Just create the full U, V, W arrays of shape (rank, dim) and fill them.
    
    total_rank = 98
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    def schoolbook(n_b, m_b, p_b):
        U_b = np.zeros((n_b * m_b, n_b * m_b * p_b), dtype=int)
        V_b = np.zeros((m_b * p_b, n_b * m_b * p_b), dtype=int)
        W_b = np.zeros((p_b * n_b, n_b * m_b * p_b), dtype=int)
        c = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j, c] = 1
                    V_b[j * p_b + k, c] = 1
                    W_b[k * n_b + i, c] = 1
                    c += 1
        return U_b, V_b, W_b

    def strassen(n_b, m_b, p_b):
        # Only for 2x2x2
        if n_b != 2 or m_b != 2 or p_b != 2:
            return schoolbook(n_b, m_b, p_b)
            
        U_b = np.zeros((4, 7), dtype=int)
        V_b = np.zeros((4, 7), dtype=int)
        W_b = np.zeros((4, 7), dtype=int)
        
        # Strassen factors
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U indices: 0:00, 1:01, 2:10, 3:11
        # V indices: 0:00, 1:01, 2:10, 3:11
        # W indices: 0:00, 1:01, 2:10, 3:11
        
        # 1
        U_b[0, 0] += 1; U_b[3, 0] += 1
        V_b[0, 0] += 1; V_b[3, 0] += 1
        W_b[0, 0] += 1; W_b[1, 0] += 1; W_b[2, 0] += 1; W_b[3, 0] += 1
        # C11 += M1, C12 -= M1, C21 += M1, C22 -= M1
        # Wait, standard Strassen:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 - M7
        
        # Let's just use the standard coefficients.
        
        # Reset
        U_b = np.zeros((4, 7), dtype=int)
        V_b = np.zeros((4, 7), dtype=int)
        W_b = np.zeros((4, 7), dtype=int)
        
        # M1
        U_b[0, 0] = 1; U_b[3, 0] = 1
        V_b[0, 0] = 1; V_b[3, 0] = 1
        W_b[0, 0] = 1; W_b[1, 0] = -1; W_b[2, 0] = 1; W_b[3, 0] = -1
        
        # M2
        U_b[2, 1] = 1; U_b[3, 1] = 1
        V_b[0, 1] = 1
        W_b[2, 1] = 1; W_b[3, 1] = 1
        
        # M3
        U_b[0, 2] = 1
        V_b[2, 2] = 1; V_b[3, 2] = -1
        W_b[1, 2] = 1; W_b[3, 2] = -1
        
        # M4
        U_b[3, 3] = 1
        V_b[1, 3] = 1; V_b[0, 3] = -1
        W_b[1, 3] = -1; W_b[2, 3] = 1
        
        # M5
        U_b[0, 4] = 1; U_b[1, 4] = 1
        V_b[3, 4] = 1
        W_b[0, 4] = 1; W_b[1, 4] = 1
        
        # M6
        U_b[2, 5] = 1; U_b[0, 5] = -1
        V_b[0, 5] = 1; V_b[1, 5] = 1
        W_b[3, 5] = 1
        
        # M7
        U_b[1, 6] = 1; U_b[3, 6] = -1
        V_b[2, 6] = 1; V_b[3, 6] = 1
        W_b[0, 6] = -1; W_b[3, 6] = -1
        
        return U_b, V_b, W_b

    def add_factors(U_b, V_b, W_b, i_off, j_off, k_off):
        nonlocal col
        rank_b = U_b.shape[1]
        for r in range(rank_b):
            for i in range(U_b.shape[0]):
                if U_b[i, r] != 0:
                    U[i + i_off, col + r] = U_b[i, r]
            for j in range(V_b.shape[0]):
                if V_b[j, r] != 0:
                    V[j + j_off, col + r] = V_b[j, r]
            for k in range(W_b.shape[0]):
                if W_b[k, r] != 0:
                    W[k + k_off, col + r] = W_b[k, r]
        col += rank_b

    # Block 1: A11 (2x2) * B11 (2x2) -> C11 part
    # i_off=0, j_off=0 (for U)
    # j_off=0, k_off=0 (for V)
    # k_off=0, i_off=0 (for W)
    U1, V1, W1 = strassen(2, 2, 2)
    add_factors(U1, V1, W1, 0, 0, 0)
    
    # Block 2: A12 (2x3) * B21 (3x2) -> C12 part? No, C11 part?
    # A12: i in 0,1; j in 2,3,4.
    # B21: j in 2,3,4; k in 0,1.
    # Result: i in 0,1; k in 0,1.
    # U offset: i=0, j=2.
    # V offset: j=2, k=0.
    # W offset: k=0, i=0.
    U2, V2, W2 = schoolbook(2, 3, 2)
    add_factors(U2, V2, W2, 0, 2, 0)
    
    # Block 3: A11 (2x2) * B12 (2x3) -> C12 part
    # A11: i in 0,1; j in 0,1.
    # B12: j in 0,1; k in 2,3,4.
    # Result: i in 0,1; k in 2,3,4.
    # U offset: i=0, j=0.
    # V offset: j=0, k=2.
    # W offset: k=2, i=0.
    U3, V3, W3 = schoolbook(2, 2, 3)
    add_factors(U3, V3, W3, 0, 0, 2)
    
    # Block 4: A12 (2x3) * B22 (3x3) -> C22 part? No, C12 part?
    # A12: i in 0,1; j in 2,3,4.
    # B22: j in 2,3,4; k in 2,3,4.
    # Result: i in 0,1; k in 2,3,4.
    # U offset: i=0, j=2.
    # V offset: j=2, k=2.
    # W offset: k=2, i=0.
    U4, V4, W4 = schoolbook(2, 3, 3)
    add_factors(U4, V4, W4, 0, 2, 2)
    
    # Block 5: A21 (2x2) * B11 (2x2) -> C21 part
    # A21: i in 2,3; j in 0,1.
    # B11: j in 0,1; k in 0,1.
    # Result: i in 2,3; k in 0,1.
    # U offset: i=2, j=0.
    # V offset: j=0, k=0.
    # W offset: k=0, i=2.
    U5, V5, W5 = strassen(2, 2, 2)
    add_factors(U5, V5, W5, 2, 0, 0)
    
    # Block 6: A22 (2x3) * B21 (3x2) -> C22 part? No, C21 part?
    # A22: i in 2,3; j in 2,3,4.
    # B21: j in 2,3,4; k in 0,1.
    # Result: i in 2,3; k in 0,1.
    # U offset: i=2, j=2.
    # V offset: j=2, k=0.
    # W offset: k=0, i=2.
    U6, V6, W6 = schoolbook(2, 3, 2)
    add_factors(U6, V6, W6, 2, 2, 0)
    
    # Block 7: A21 (2x2) * B12 (2x3) -> C22 part? No, C21 part?
    # A21: i in 2,3; j in 0,1.
    # B12: j in 0,1; k in 2,3,4.
    # Result: i in 2,3; k in 2,3,4.
    # U offset: i=2, j=0.
    # V offset: j=0, k=2.
    # W offset: k=2, i=2.
    U7, V7, W7 = schoolbook(2, 2, 3)
    add_factors(U7, V7, W7, 2, 0, 2)
    
    # Block 8: A22 (2x3) * B22 (3x3) -> C22 part
    # A22: i in 2,3; j in 2,3,4.
    # B22: j in 2,3,4; k in 2,3,4.
    # Result: i in 2,3; k in 2,3,4.
    # U offset: i=2, j=2.
    # V offset: j=2, k=2.
    # W offset: k=2, i=2.
    U8, V8, W8 = schoolbook(2, 3, 3)
    add_factors(U8, V8, W8, 2, 2, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
