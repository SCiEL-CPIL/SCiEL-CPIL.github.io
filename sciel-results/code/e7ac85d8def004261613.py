"""Optimized matrix multiplication decomposition for 2x5 x 5x5 using Strassen blocks."""
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5  # A is n x m, B is m x p, C is n x p
    # U index: i*m + j = i*5 + j (i=0,1; j=0..4)
    # V index: j*p + k = j*5 + k (j=0..4; k=0..4)
    # W index: k*n + i = k*2 + i (k=0..4; i=0,1)
    
    terms = []  # Each term: (u_list, v_list, w_list)
    
    def add_term(u_list, v_list, w_list):
        terms.append((u_list, v_list, w_list))
    
    def add_strassen_2x2x2(a_col_off, b_row_off, b_col_off, c_col_off):
        """Strassen for A[:, a_col_off:a_col_off+2] * B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]
        -> C[:, c_col_off:c_col_off+2]
        A rows are always 0,1 (since n=2)."""
        # A elements: A[i, a_col_off] and A[i, a_col_off+1] for i=0,1
        # B elements: B[b_row_off, b_col_off], B[b_row_off, b_col_off+1], B[b_row_off+1, b_col_off], B[b_row_off+1, b_col_off+1]
        # C elements: C[i, c_col_off] and C[i, c_col_off+1] for i=0,1
        
        # M1 = (A11 + A22)(B11 + B22)
        u_m1 = [(0*5 + a_col_off, 1), (1*5 + (a_col_off+1), 1)]
        v_m1 = [(b_row_off*5 + b_col_off, 1), ((b_row_off+1)*5 + (b_col_off+1), 1)]
        w_m1 = [(c_col_off*2 + 0, 1), ((c_col_off+1)*2 + 0, 1),
                 (c_col_off*2 + 1, 1), ((c_col_off+1)*2 + 1, 1)]
        add_term(u_m1, v_m1, w_m1)
        
        # M2 = (A21 + A22) B11
        u_m2 = [(1*5 + a_col_off, 1), (1*5 + (a_col_off+1), 1)]
        v_m2 = [(b_row_off*5 + b_col_off, 1)]
        w_m2 = [(c_col_off*2 + 1, 1), ((c_col_off+1)*2 + 1, -1)]
        add_term(u_m2, v_m2, w_m2)
        
        # M3 = A11 (B12 - B22)
        u_m3 = [(0*5 + a_col_off, 1)]
        v_m3 = [(b_row_off*5 + (b_col_off+1), 1), ((b_row_off+1)*5 + (b_col_off+1), -1)]
        w_m3 = [((c_col_off+1)*2 + 0, 1), ((c_col_off+1)*2 + 1, 1)]
        add_term(u_m3, v_m3, w_m3)
        
        # M4 = A22 (B21 - B11)
        u_m4 = [(1*5 + (a_col_off+1), 1)]
        v_m4 = [((b_row_off+1)*5 + b_col_off, 1), (b_row_off*5 + b_col_off, -1)]
        w_m4 = [(c_col_off*2 + 0, 1), (c_col_off*2 + 1, 1)]
        add_term(u_m4, v_m4, w_m4)
        
        # M5 = (A11 + A12) B22
        u_m5 = [(0*5 + a_col_off, 1), (0*5 + (a_col_off+1), 1)]
        v_m5 = [((b_row_off+1)*5 + (b_col_off+1), 1)]
        w_m5 = [(c_col_off*2 + 0, -1), ((c_col_off+1)*2 + 0, 1),
                 (c_col_off*2 + 1, -1), ((c_col_off+1)*2 + 1, 1)]
        add_term(u_m5, v_m5, w_m5)
        
        # M6 = (A21 - A11)(B11 + B12)
        u_m6 = [(1*5 + a_col_off, 1), (0*5 + a_col_off, -1)]
        v_m6 = [(b_row_off*5 + b_col_off, 1), (b_row_off*5 + (b_col_off+1), 1)]
        w_m6 = [((c_col_off+1)*2 + 0, 1), ((c_col_off+1)*2 + 1, 1)]
        add_term(u_m6, v_m6, w_m6)
        
        # M7 = (A12 - A22)(B21 + B22)
        u_m7 = [(0*5 + (a_col_off+1), 1), (1*5 + (a_col_off+1), -1)]
        v_m7 = [((b_row_off+1)*5 + b_col_off, 1), ((b_row_off+1)*5 + (b_col_off+1), 1)]
        w_m7 = [(c_col_off*2 + 0, 1), (c_col_off*2 + 1, 1)]
        add_term(u_m7, v_m7, w_m7)
    
    # Four Strassen blocks:
    # A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    add_strassen_2x2x2(0, 0, 0, 0)
    # A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    add_strassen_2x2x2(0, 0, 2, 2)
    # A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    add_strassen_2x2x2(2, 2, 0, 0)
    # A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    add_strassen_2x2x2(2, 2, 2, 2)
    
    # Remaining schoolbook terms (18 total):
    # A[:, 0:2] * B[0:2, 4] -> C[:, 4]  (4 terms)
    for i in range(2):
        for j in range(2):
            add_term([(i*5 + j, 1)], [(j*5 + 4, 1)], [(4*2 + i, 1)])
    
    # A[:, 2:4] * B[2:4, 4] -> C[:, 4]  (4 terms)
    for i in range(2):
        for j in range(2, 4):
            add_term([(i*5 + j, 1)], [(j*5 + 4, 1)], [(4*2 + i, 1)])
    
    # A[:, 4] * B[4, 0:2] -> C[:, 0:2]  (4 terms)
    for i in range(2):
        for k in range(2):
            add_term([(i*5 + 4, 1)], [(4*5 + k, 1)], [(k*2 + i, 1)])
    
    # A[:, 4] * B[4, 2:4] -> C[:, 2:4]  (4 terms)
    for i in range(2):
        for k in range(2, 4):
            add_term([(i*5 + 4, 1)], [(4*5 + k, 1)], [(k*2 + i, 1)])
    
    # A[:, 4] * B[4, 4] -> C[:, 4]  (2 terms)
    for i in range(2):
        add_term([(i*5 + 4, 1)], [(4*5 + 4, 1)], [(4*2 + i, 1)])
    
    rank = len(terms)
    assert rank == 46, f"Expected rank 46, got {rank}"
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]  # 10 x rank
    V = [[0] * rank for _ in range(m * p)]  # 25 x rank
    W = [[0] * rank for _ in range(p * n)]  # 10 x rank
    
    for col, (u_list, v_list, w_list) in enumerate(terms):
        for idx, coeff in u_list:
            U[idx][col] += coeff
        for idx, coeff in v_list:
            V[idx][col] += coeff
        for idx, coeff in w_list:
            W[idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
