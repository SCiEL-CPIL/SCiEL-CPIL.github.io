import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    # Total rank calculation
    rank_strassen_444 = 7
    rank_strassen_442 = 28
    rank_update = 1
    total_rank = rank_strassen_444 + rank_strassen_442 + rank_update
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # Helper to add a rank-1 term (or rank-k term) to U, V, W
    # u_indices: list of (i, j) for A
    # v_indices: list of (j, k) for B
    # w_indices: list of (k, i) for C
    # coeffs_u, coeffs_v, coeffs_w: values
    # This is for a single rank unit.
    
    def add_term(u_map, v_map, w_map, col):
        for (i, j), val in u_map.items():
            U[i * m + j, col] += val
        for (j, k), val in v_map.items():
            V[j * p + k, col] += val
        for (k, i), val in w_map.items():
            W[k * n + i, col] += val

    # --- Part 1: Strassen 4x4x4 ---
    # A1 is A[0:4, 0:4]
    # B11 is B[0:4, 0:4]
    # C1 is C[0:4, 0:4]
    
    # Strassen 7 multiplications
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Block indices in 4x4 matrix (0:2, 2:4)
    # A11: i 0-1, j 0-1
    # A12: i 0-1, j 2-3
    # A21: i 2-3, j 0-1
    # A22: i 2-3, j 2-3
    
    # B11: j 0-1, k 0-1
    # B12: j 0-1, k 2-3
    # B21: j 2-3, k 0-1
    # B22: j 2-3, k 2-3
    
    # C11: k 0-1, i 0-1
    # C12: k 0-1, i 2-3
    # C21: k 2-3, i 0-1
    # C22: k 2-3, i 2-3
    
    def get_block_indices(start_i, end_i, start_j, end_j):
        return [(i, j) for i in range(start_i, end_i) for j in range(start_j, end_j)]
        
    def get_block_indices_b(start_j, end_j, start_k, end_k):
        return [(j, k) for j in range(start_j, end_j) for k in range(start_k, end_k)]
        
    def get_block_indices_c(start_k, end_k, start_i, end_i):
        return [(k, i) for k in range(start_k, end_k) for i in range(start_i, end_i)]

    col = 0
    
    # M1
    u_map = {}
    for i, j in get_block_indices(0, 2, 0, 2): u_map[(i, j)] = 1
    for i, j in get_block_indices(2, 4, 2, 4): u_map[(i, j)] = 1
    v_map = {}
    for j, k in get_block_indices_b(0, 2, 0, 2): v_map[(j, k)] = 1
    for j, k in get_block_indices_b(2, 4, 2, 4): v_map[(j, k)] = 1
    w_map = {}
    # C11 = M1...
    for k, i in get_block_indices_c(0, 2, 0, 2): w_map[(k, i)] = 1
    # C22 = M1...
    for k, i in get_block_indices_c(2, 4, 2, 4): w_map[(k, i)] = 1
    add_term(u_map, v_map, w_map, col)
    col += 1
    
    # M2
    u_map = {}
    for i, j in get_block_indices(2, 4, 0, 2): u_map[(i, j)] = 1
    for i, j in get_block_indices(2, 4, 2, 4): u_map[(i, j)] = 1
    v_map = {}
    for j, k in get_block_indices_b(0, 2, 0, 2): v_map[(j, k)] = 1
    w_map = {}
    # C21 = M2...
    for k, i in get_block_indices_c(2, 4, 0, 2): w_map[(k, i)] = 1
    add_term(u_map, v_map, w_map, col)
    col += 1
    
    # M3
    u_map = {}
    for i, j in get_block_indices(0, 2, 0, 2): u_map[(i, j)] = 1
    v_map = {}
    for j, k in get_block_indices_b(0, 2, 2, 4): v_map[(j, k)] = 1
    for j, k in get_block_indices_b(2, 4, 2, 4): v_map[(j, k)] = -1
    w_map = {}
    # C12 = M3...
    for k, i in get_block_indices_c(0, 2, 2, 4): w_map[(k, i)] = 1
    # C22 = -M3...
    for k, i in get_block_indices_c(2, 4, 2, 4): w_map[(k, i)] = -1
    add_term(u_map, v_map, w_map, col)
    col += 1
    
    # M4
    u_map = {}
    for i, j in get_block_indices(2, 4, 2, 4): u_map[(i, j)] = 1
    v_map = {}
    for j, k in get_block_indices_b(2, 4, 0, 2): v_map[(j, k)] = 1
    for j, k in get_block_indices_b(0, 2, 0, 2): v_map[(j, k)] = -1
    w_map = {}
    # C11 = -M4...
    for k, i in get_block_indices_c(0, 2, 0, 2): w_map[(k, i)] = -1
    # C21 = M4...
    for k, i in get_block_indices_c(2, 4, 0, 2): w_map[(k, i)] = 1
    add_term(u_map, v_map, w_map, col)
    col += 1
    
    # M5
    u_map = {}
    for i, j in get_block_indices(0, 2, 0, 2): u_map[(i, j)] = 1
    for i, j in get_block_indices(0, 2, 2, 4): u_map[(i, j)] = 1
    v_map = {}
    for j, k in get_block_indices_b(2, 4, 2, 4): v_map[(j, k)] = 1
    w_map = {}
    # C11 = M5...
    for k, i in get_block_indices_c(0, 2, 0, 2): w_map[(k, i)] = 1
    add_term(u_map, v_map, w_map, col)
    col += 1
    
    # M6
    u_map = {}
    for i, j in get_block_indices(2, 4, 0, 2): u_map[(i, j)] = 1
    for i, j in get_block_indices(0, 2, 0, 2): u_map[(i, j)] = -1
    v_map = {}
    for j, k in get_block_indices_b(0, 2, 0, 2): v_map[(j, k)] = 1
    for j, k in get_block_indices_b(0, 2, 2, 4): v_map[(j, k)] = 1
    w_map = {}
    # C22 = M6...
    for k, i in get_block_indices_c(2, 4, 2, 4): w_map[(k, i)] = 1
    add_term(u_map, v_map, w_map, col)
    col += 1
    
    # M7
    u_map = {}
    for i, j in get_block_indices(0, 2, 2, 4): u_map[(i, j)] = 1
    for i, j in get_block_indices(2, 4, 2, 4): u_map[(i, j)] = -1
    v_map = {}
    for j, k in get_block_indices_b(2, 4, 0, 2): v_map[(j, k)] = 1
    for j, k in get_block_indices_b(2, 4, 2, 4): v_map[(j, k)] = 1
    w_map = {}
    # C11 = -M7...
    for k, i in get_block_indices_c(0, 2, 0, 2): w_map[(k, i)] = -1
    # C12 = M7...
    for k, i in get_block_indices_c(0, 2, 2, 4): w_map[(k, i)] = 1
    # C22 = M7...
    for k, i in get_block_indices_c(2, 4, 2, 4): w_map[(k, i)] = 1
    add_term(u_map, v_map, w_map, col)
    col += 1
    
    # --- Part 2: 4x4x2 using 4x 2x2x2 Strassen ---
    # We need a helper to add a 2x2x2 Strassen decomposition.
    # It takes offsets for i, j, k.
    
    def add_strassen_222(offset_i, offset_j, offset_k, start_col):
        # A block is 2x2 at (offset_i, offset_j)
        # B block is 2x2 at (offset_j, offset_k)
        # C block is 2x2 at (offset_k, offset_i)
        
        # M1 = (A11+A22)(B11+B22)
        u_map = {}
        u_map[(offset_i, offset_j)] = 1
        u_map[(offset_i+1, offset_j+1)] = 1
        v_map = {}
        v_map[(offset_j, offset_k)] = 1
        v_map[(offset_j+1, offset_k+1)] = 1
        w_map = {}
        w_map[(offset_k, offset_i)] = 1
        w_map[(offset_k+1, offset_i+1)] = 1
        add_term(u_map, v_map, w_map, start_col)
        
        # M2 = (A21+A22)B11
        u_map = {}
        u_map[(offset_i+1, offset_j)] = 1
        u_map[(offset_i+1, offset_j+1)] = 1
        v_map = {}
        v_map[(offset_j, offset_k)] = 1
        w_map = {}
        w_map[(offset_k+1, offset_i)] = 1
        add_term(u_map, v_map, w_map, start_col + 1)
        
        # M3 = A11(B12-B22)
        u_map = {}
        u_map[(offset_i, offset_j)] = 1
        v_map = {}
        v_map[(offset_j, offset_k+1)] = 1
        v_map[(offset_j+1, offset_k+1)] = -1
        w_map = {}
        w_map[(offset_k, offset_i+1)] = 1
        w_map[(offset_k+1, offset_i+1)] = -1
        add_term(u_map, v_map, w_map, start_col + 2)
        
        # M4 = A22(B21-B11)
        u_map = {}
        u_map[(offset_i+1, offset_j+1)] = 1
        v_map = {}
        v_map[(offset_j+1, offset_k)] = 1
        v_map[(offset_j, offset_k)] = -1
        w_map = {}
        w_map[(offset_k, offset_i)] = -1
        w_map[(offset_k+1, offset_i)] = 1
        add_term(u_map, v_map, w_map, start_col + 3)
        
        # M5 = (A11+A12)B22
        u_map = {}
        u_map[(offset_i, offset_j)] = 1
        u_map[(offset_i, offset_j+1)] = 1
        v_map = {}
        v_map[(offset_j+1, offset_k+1)] = 1
        w_map = {}
        w_map[(offset_k, offset_i)] = 1
        add_term(u_map, v_map, w_map, start_col + 4)
        
        # M6 = (A21-A11)(B11+B12)
        u_map = {}
        u_map[(offset_i+1, offset_j)] = 1
        u_map[(offset_i, offset_j)] = -1
        v_map = {}
        v_map[(offset_j, offset_k)] = 1
        v_map[(offset_j, offset_k+1)] = 1
        w_map = {}
        w_map[(offset_k+1, offset_i+1)] = 1
        add_term(u_map, v_map, w_map, start_col + 5)
        
        # M7 = (A12-A22)(B21+B22)
        u_map = {}
        u_map[(offset_i, offset_j+1)] = 1
        u_map[(offset_i+1, offset_j+1)] = -1
        v_map = {}
        v_map[(offset_j+1, offset_k)] = 1
        v_map[(offset_j+1, offset_k+1)] = 1
        w_map = {}
        w_map[(offset_k, offset_i)] = -1
        w_map[(offset_k, offset_i+1)] = 1
        w_map[(offset_k+1, offset_i+1)] = 1
        add_term(u_map, v_map, w_map, start_col + 6)
        
        return start_col + 7

    # 1. A11 * B_top -> C_top_left?
    # A11 is i 0-1, j 0-1.
    # B_top is j 0-1, k 4-5 (cols 4,5 of B).
    # C_top is i 0-1, k 4-5.
    # Wait, C_top is rows 0-1 of C. C is 4x6. C_top is 2x6?
    # No, we split C2 (4x2) into C_top (2x2) and C_bot (2x2).
    # C_top corresponds to i 0-1, k 4-5.
    # So offset_i=0, offset_k=4.
    # A11 is i 0-1, j 0-1. offset_i=0, offset_j=0.
    # B_top is j 0-1, k 4-5. offset_j=0, offset_k=4.
    col = add_strassen_222(0, 0, 4, col)
    
    # 2. A12 * B_bot -> C_top_right?
    # A12 is i 0-1, j 2-3. offset_i=0, offset_j=2.
    # B_bot is j 2-3, k 4-5. offset_j=2, offset_k=4.
    # C_top is i 0-1, k 4-5. offset_i=0, offset_k=4.
    col = add_strassen_222(0, 2, 4, col)
    
    # 3. A21 * B_top -> C_bot_left?
    # A21 is i 2-3, j 0-1. offset_i=2, offset_j=0.
    # B_top is j 0-1, k 4-5. offset_j=0, offset_k=4.
    # C_bot is i 2-3, k 4-5. offset_i=2, offset_k=4.
    col = add_strassen_222(2, 0, 4, col)
    
    # 4. A22 * B_bot -> C_bot_right?
    # A22 is i 2-3, j 2-3. offset_i=2, offset_j=2.
    # B_bot is j 2-3, k 4-5. offset_j=2, offset_k=4.
    # C_bot is i 2-3, k 4-5. offset_i=2, offset_k=4.
    col = add_strassen_222(2, 2, 4, col)
    
    # --- Part 3: Rank 1 Update ---
    # a is A[:, 4]. b is B[4, :].
    # U[i, 4] = 1 for all i
    # V[4, k] = 1 for all k
    # W[k, i] = 1 for all k, i
    u_map = {}
    for i in range(n):
        u_map[(i, 4)] = 1
    v_map = {}
    for k in range(p):
        v_map[(4, k)] = 1
    w_map = {}
    for k in range(p):
        for i in range(n):
            w_map[(k, i)] = 1
    add_term(u_map, v_map, w_map, col)
    col += 1
    
    return {"U": U, "V": V, "W": W}
