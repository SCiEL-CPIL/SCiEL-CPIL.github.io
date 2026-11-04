import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    def add_rank1_term(u_sparse, v_sparse, w_sparse):
        """
        Adds a single rank-1 term defined by sparse vectors.
        u_sparse, v_sparse, w_sparse: dict {global_idx: coeff}
        """
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for idx, coeff in u_sparse.items():
            u_vec[idx] = coeff
        for idx, coeff in v_sparse.items():
            v_vec[idx] = coeff
        for idx, coeff in w_sparse.items():
            w_vec[idx] = coeff
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen_block(a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset):
        # M1: (A11+A22)(B11+B22) -> C11, C22
        u = {u_idx(a_i_offset+0, a_j_offset+0): 1, u_idx(a_i_offset+1, a_j_offset+1): 1}
        v = {v_idx(b_j_offset+0, b_k_offset+0): 1, v_idx(b_j_offset+1, b_k_offset+1): 1}
        w = {w_idx(c_k_offset+0, c_i_offset+0): 1, w_idx(c_k_offset+1, c_i_offset+1): 1}
        add_rank1_term(u, v, w)
        
        # M2: (A21+A22)B11 -> C21
        u = {u_idx(a_i_offset+1, a_j_offset+0): 1, u_idx(a_i_offset+1, a_j_offset+1): 1}
        v = {v_idx(b_j_offset+0, b_k_offset+0): 1}
        w = {w_idx(c_k_offset+0, c_i_offset+1): 1}
        add_rank1_term(u, v, w)
        
        # M3: A11(B12-B22) -> C12
        u = {u_idx(a_i_offset+0, a_j_offset+0): 1}
        v = {v_idx(b_j_offset+0, b_k_offset+1): 1, v_idx(b_j_offset+1, b_k_offset+1): -1}
        w = {w_idx(c_k_offset+1, c_i_offset+0): 1}
        add_rank1_term(u, v, w)
        
        # M4: A22(B21-B11) -> C11, C21
        u = {u_idx(a_i_offset+1, a_j_offset+1): 1}
        v = {v_idx(b_j_offset+1, b_k_offset+0): 1, v_idx(b_j_offset+0, b_k_offset+0): -1}
        w = {w_idx(c_k_offset+0, c_i_offset+0): 1, w_idx(c_k_offset+0, c_i_offset+1): 1}
        add_rank1_term(u, v, w)
        
        # M5: (A11+A12)B22 -> C11 (-1), C12 (1)
        u = {u_idx(a_i_offset+0, a_j_offset+0): 1, u_idx(a_i_offset+0, a_j_offset+1): 1}
        v = {v_idx(b_j_offset+1, b_k_offset+1): 1}
        w = {w_idx(c_k_offset+0, c_i_offset+0): -1, w_idx(c_k_offset+1, c_i_offset+0): 1}
        add_rank1_term(u, v, w)
        
        # M6: (A21-A11)(B11+B12) -> C22
        u = {u_idx(a_i_offset+1, a_j_offset+0): 1, u_idx(a_i_offset+0, a_j_offset+0): -1}
        v = {v_idx(b_j_offset+0, b_k_offset+0): 1, v_idx(b_j_offset+0, b_k_offset+1): 1}
        w = {w_idx(c_k_offset+1, c_i_offset+1): 1}
        add_rank1_term(u, v, w)
        
        # M7: (A12-A22)(B21+B22) -> C11
        u = {u_idx(a_i_offset+0, a_j_offset+1): 1, u_idx(a_i_offset+1, a_j_offset+1): -1}
        v = {v_idx(b_j_offset+1, b_k_offset+0): 1, v_idx(b_j_offset+1, b_k_offset+1): 1}
        w = {w_idx(c_k_offset+0, c_i_offset+0): 1}
        add_rank1_term(u, v, w)

    def add_schoolbook_block(a_i_start, a_i_end, a_j_start, a_j_end, b_j_start, b_j_end, b_k_start, b_k_end, c_i_start, c_i_end, c_k_start, c_k_end):
        for i in range(a_i_start, a_i_end):
            for j in range(a_j_start, a_j_end):
                for k in range(b_k_start, b_k_end):
                    u = {u_idx(i, j): 1}
                    v = {v_idx(j, k): 1}
                    w = {w_idx(k, i): 1}
                    add_rank1_term(u, v, w)

    # Block 1: A_1 (i=0..2, j=0..1) * B_11 (j=0..1, k=0..1) -> C (i=0..2, k=0..1)
    # Strassen for i=0,1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    # Schoolbook for i=2
    add_schoolbook_block(2, 3, 0, 2, 0, 2, 0, 2, 2, 3, 0, 2)

    # Block 2: A_1 (i=0..2, j=0..1) * B_12 (j=0..1, k=2..3) -> C (i=0..2, k=2..3)
    # Strassen for i=0,1
    add_strassen_block(0, 0, 0, 2, 0, 2)
    # Schoolbook for i=2
    add_schoolbook_block(2, 3, 0, 2, 0, 2, 2, 4, 2, 3, 2, 4)

    # Block 3: A_1 (i=0..2, j=0..1) * B_13 (j=0..1, k=4) -> C (i=0..2, k=4)
    # Schoolbook for all i
    add_schoolbook_block(0, 3, 0, 2, 0, 2, 4, 5, 0, 3, 4, 5)

    # Block 4: A_2 (i=0..2, j=2..3) * B_21 (j=2..3, k=0..1) -> C (i=0..2, k=0..1)
    # Strassen for i=0,1
    add_strassen_block(0, 2, 2, 0, 0, 0)
    # Schoolbook for i=2
    add_schoolbook_block(2, 3, 2, 4, 2, 4, 0, 2, 2, 3, 0, 2)

    # Block 5: A_2 (i=0..2, j=2..3) * B_22 (j=2..3, k=2..3) -> C (i=0..2, k=2..3)
    # Strassen for i=0,1
    add_strassen_block(0, 2, 2, 2, 0, 2)
    # Schoolbook for i=2
    add_schoolbook_block(2, 3, 2, 4, 2, 4, 2, 4, 2, 3, 2, 4)

    # Block 6: A_2 (i=0..2, j=2..3) * B_23 (j=2..3, k=4) -> C (i=0..2, k=4)
    # Schoolbook for all i
    add_schoolbook_block(0, 3, 2, 4, 2, 4, 4, 5, 0, 3, 4, 5)

    # Block 7: A_3 (i=0..2, j=4) * B_31 (j=4, k=0..1) -> C (i=0..2, k=0..1)
    # Schoolbook for all i
    add_schoolbook_block(0, 3, 4, 5, 4, 5, 0, 2, 0, 3, 0, 2)

    # Block 8: A_3 (i=0..2, j=4) * B_32 (j=4, k=2..3) -> C (i=0..2, k=2..3)
    # Schoolbook for all i
    add_schoolbook_block(0, 3, 4, 5, 4, 5, 2, 4, 0, 3, 2, 4)

    # Block 9: A_3 (i=0..2, j=4) * B_33 (j=4, k=4) -> C (i=0..2, k=4)
    # Schoolbook for all i
    add_schoolbook_block(0, 3, 4, 5, 4, 5, 4, 5, 0, 3, 4, 5)

    # Assemble U, V, W
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for i in range(m * p):
            V[i][r] = V_cols[r][i]
        for i in range(p * n):
            W[i][r] = W_cols[r][i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
