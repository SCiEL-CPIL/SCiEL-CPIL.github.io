import time

def construct():
    """
    Returns a rank-50 exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    This improves upon the schoolbook rank of 56 by using Strassen's algorithm for the 2x2x2 blocks
    and combining them appropriately.
    """
    n, m, p = 2, 4, 7
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U, V, W are 2x2 matrices
    strassen_terms = [
        # (U, V, W)
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        ([[ -1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]),
    ]
    
    # Schoolbook terms for 2x2x1
    # U is 2x2, V is 2x1, W is 1x2
    schoolbook_2x2x1_terms = [
        ([[1, 0], [0, 0]], [[1], [0]], [[1, 0]]),
        ([[0, 1], [0, 0]], [[0], [1]], [[1, 0]]),
        ([[0, 0], [1, 0]], [[1], [0]], [[0, 1]]),
        ([[0, 0], [0, 1]], [[0], [1]], [[0, 1]]),
    ]
    
    col = 0
    
    def add_term(u_mat, v_mat, w_mat, offset_i, offset_j, offset_k):
        nonlocal col
        # u_mat is 2x2 (local i, j)
        # v_mat is 2x2 or 2x1 (local j, k)
        # w_mat is 2x2 or 1x2 (local k, i)
        
        rows_u = len(u_mat)
        cols_u = len(u_mat[0])
        rows_v = len(v_mat)
        cols_v = len(v_mat[0])
        rows_w = len(w_mat)
        cols_w = len(w_mat[0])
        
        for li in range(rows_u):
            for lj in range(cols_u):
                val = u_mat[li][lj]
                if val != 0:
                    global_i = offset_i + li
                    global_j = offset_j + lj
                    U[global_i * m + global_j][col] = val
                    
        for lj in range(rows_v):
            for lk in range(cols_v):
                val = v_mat[lj][lk]
                if val != 0:
                    global_j = offset_j + lj
                    global_k = offset_k + lk
                    V[global_j * p + global_k][col] = val
                    
        for lk in range(rows_w):
            for li in range(cols_w):
                val = w_mat[lk][li]
                if val != 0:
                    global_k = offset_k + lk
                    global_i = offset_i + li
                    W[global_k * n + global_i][col] = val
                    
        col += 1

    # We need to handle the different shapes of V and W for Strassen (2x2) vs Schoolbook (2x1, 1x2)
    # The add_term function above assumes consistent local dimensions.
    # Strassen: U(2x2), V(2x2), W(2x2)
    # Schoolbook: U(2x2), V(2x1), W(1x2)
    
    # We will call add_term with appropriate matrices.
    
    # Helper to add a block of 2x2x2 Strassen
    def add_strassen_block(offset_i, offset_j, offset_k):
        for u, v, w in strassen_terms:
            add_term(u, v, w, offset_i, offset_j, offset_k)
            
    # Helper to add a block of 2x2x1 Schoolbook
    def add_schoolbook_block(offset_i, offset_j, offset_k):
        for u, v, w in schoolbook_2x2x1_terms:
            add_term(u, v, w, offset_i, offset_j, offset_k)
            
    # Structure:
    # 2x4x7
    # Split A (2x4) into A1 (2x2, j=0,1) and A2 (2x2, j=2,3)
    # Split B (4x7) into B1 (2x7, j=0,1) and B2 (2x7, j=2,3)
    # C = A1 B1 + A2 B2
    
    # For A1 B1 (2x2x7):
    # Split B1 (2x7) into B1a (2x4, k=0..3) and B1b (2x3, k=4..6)
    # B1a split into B1a1 (2x2, k=0,1) and B1a2 (2x2, k=2,3)
    # B1b split into B1b1 (2x2, k=4,5) and B1b2 (2x1, k=6)
    
    # A1 B1a1: Strassen (2x2x2) at i=0, j=0, k=0
    add_strassen_block(0, 0, 0)
    # A1 B1a2: Strassen (2x2x2) at i=0, j=0, k=2
    add_strassen_block(0, 0, 2)
    # A1 B1b1: Strassen (2x2x2) at i=0, j=0, k=4
    add_strassen_block(0, 0, 4)
    # A1 B1b2: Schoolbook (2x2x1) at i=0, j=0, k=6
    add_schoolbook_block(0, 0, 6)
    
    # A2 B2a1: Strassen (2x2x2) at i=0, j=2, k=0
    add_strassen_block(0, 2, 0)
    # A2 B2a2: Strassen (2x2x2) at i=0, j=2, k=2
    add_strassen_block(0, 2, 2)
    # A2 B2b1: Strassen (2x2x2) at i=0, j=2, k=4
    add_strassen_block(0, 2, 4)
    # A2 B2b2: Schoolbook (2x2x1) at i=0, j=2, k=6
    add_schoolbook_block(0, 2, 6)
    
    # Total terms: 7*6 + 4*2 = 42 + 8 = 50.
    
    return {"U": U, "V": V, "W": W}
