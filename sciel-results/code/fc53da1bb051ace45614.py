import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    terms = []
    
    def add_term(u_terms, v_terms, w_terms):
        terms.append((u_terms, v_terms, w_terms))

    def add_strassen(a_idx, b_idx, c_idx):
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        add_term([(a00, 1.0), (a11, 1.0)], 
                 [(b00, 1.0), (b11, 1.0)], 
                 [(c00, 1.0), (c11, 1.0)])
                 
        add_term([(a10, 1.0), (a11, 1.0)], 
                 [(b00, 1.0)], 
                 [(c10, 1.0)])
                 
        add_term([(a00, 1.0)], 
                 [(b01, 1.0), (b11, -1.0)], 
                 [(c01, 1.0), (c11, 1.0)])
                 
        add_term([(a11, 1.0)], 
                 [(b10, 1.0), (b00, -1.0)], 
                 [(c00, 1.0), (c10, 1.0)])
                 
        add_term([(a00, 1.0), (a01, 1.0)], 
                 [(b11, 1.0)], 
                 [(c00, -1.0), (c01, 1.0)])
                 
        add_term([(a10, 1.0), (a00, -1.0)], 
                 [(b00, 1.0), (b01, 1.0)], 
                 [(c11, 1.0)])
                 
        add_term([(a01, 1.0), (a11, -1.0)], 
                 [(b10, 1.0), (b11, 1.0)], 
                 [(c00, 1.0)])
                 
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Term 1: 4 Strassen blocks
    a11 = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    a12 = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    
    b11 = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    b12 = [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)]
    b21 = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)]
    b22 = [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)]
    
    c01 = [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)] # C[0,0], C[0,1], C[1,0], C[1,1]
    c23 = [w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)] # C[0,2], C[0,3], C[1,2], C[1,3]
    
    add_strassen(a11, b11, c01)
    add_strassen(a11, b12, c23)
    add_strassen(a12, b21, c01)
    add_strassen(a12, b22, c23)
    
    # Term 2: A_4 * B_42
    for k in [4, 5]:
        for i in [0, 1]:
            for j in [0, 1, 2, 3]:
                add_term([(u_idx(i, j), 1.0)], 
                         [(v_idx(j, k), 1.0)], 
                         [(w_idx(k, i), 1.0)])
                         
    # Term 3: A_1 * B_14
    for k in [0, 1, 2, 3]:
        for i in [0, 1]:
            add_term([(u_idx(i, 4), 1.0)], 
                     [(v_idx(4, k), 1.0)], 
                     [(w_idx(k, i), 1.0)])
                     
    # Term 4: A_1 * B_12
    for k in [4, 5]:
        for i in [0, 1]:
            add_term([(u_idx(i, 4), 1.0)], 
                     [(v_idx(4, k), 1.0)], 
                     [(w_idx(k, i), 1.0)])
                     
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for col_idx, (u_terms, v_terms, w_terms) in enumerate(terms):
        for r, c in u_terms:
            U[r][col_idx] = c
        for r, c in v_terms:
            V[r][col_idx] = c
        for r, c in w_terms:
            W[r][col_idx] = c
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
