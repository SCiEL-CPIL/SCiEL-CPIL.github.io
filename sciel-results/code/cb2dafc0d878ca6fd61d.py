import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2x2
    # u[i, j], v[j, k], w[k, i]
    # M1: A00 (B00+B11) -> C00, C11
    strassen_terms = [
        { 'u': {(0,0): 1}, 'v': {(0,0): 1, (1,1): 1}, 'w': {(0,0): 1, (1,1): 1} }, # M1
        { 'u': {(1,0): 1, (1,1): 1}, 'v': {(0,0): 1}, 'w': {(1,0): 1, (1,1): -1} }, # M2
        { 'u': {(0,0): 1}, 'v': {(1,0): 1, (1,1): -1}, 'w': {(0,1): 1, (1,1): 1} }, # M3
        { 'u': {(1,1): 1}, 'v': {(0,1): 1, (0,0): -1}, 'w': {(0,0): 1, (1,0): 1} }, # M4
        { 'u': {(0,0): 1, (1,1): 1}, 'v': {(1,1): 1}, 'w': {(0,0): -1, (0,1): 1} }, # M5
        { 'u': {(0,1): 1, (1,1): -1}, 'v': {(1,0): 1, (1,1): 1}, 'w': {(1,1): 1} }, # M6
        { 'u': {(1,0): 1, (0,0): -1}, 'v': {(0,0): 1, (0,1): 1}, 'w': {(0,0): 1} }, # M7
    ]
    
    # Schoolbook terms for 2x2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # k is always 0 locally
    schoolbook_2x2x1_terms = [
        { 'u': {(0,0): 1}, 'v': {(0,0): 1}, 'w': {(0,0): 1} }, # A00 B0 -> C0
        { 'u': {(0,1): 1}, 'v': {(1,0): 1}, 'w': {(0,0): 1} }, # A01 B1 -> C0
        { 'u': {(1,0): 1}, 'v': {(0,0): 1}, 'w': {(1,0): 1} }, # A10 B0 -> C1
        { 'u': {(1,1): 1}, 'v': {(1,0): 1}, 'w': {(1,0): 1} }, # A11 B1 -> C1
    ]

    def add_term(u_dict, v_dict, w_dict, i_off, j_off, k_off):
        nonlocal col
        if col >= rank:
            return
        for (li, lj), val in u_dict.items():
            gi = li + i_off
            gj = lj + j_off
            U[gi * m + gj][col] += val
        for (lj, lk), val in v_dict.items():
            gj = lj + j_off
            gk = lk + k_off
            V[gj * p + gk][col] += val
        for (lk, li), val in w_dict.items():
            gk = lk + k_off
            gi = li + i_off
            W[gk * n + gi][col] += val
        col += 1

    # Block 1: A_left (j 0-1) * B_1_top (j 0-1, k 0-1)
    for term in strassen_terms:
        add_term(term['u'], term['v'], term['w'], 0, 0, 0)
        
    # Block 2: A_right (j 2-3) * B_1_bottom (j 2-3, k 0-1)
    for term in strassen_terms:
        add_term(term['u'], term['v'], term['w'], 0, 2, 0)
        
    # Block 3: A_left (j 0-1) * B_2_top (j 0-1, k 2-3)
    for term in strassen_terms:
        add_term(term['u'], term['v'], term['w'], 0, 0, 2)
        
    # Block 4: A_right (j 2-3) * B_2_bottom (j 2-3, k 2-3)
    for term in strassen_terms:
        add_term(term['u'], term['v'], term['w'], 0, 2, 2)
        
    # Block 5: A_left (j 0-1) * B_3_top (j 0-1, k 4)
    for term in schoolbook_2x2x1_terms:
        add_term(term['u'], term['v'], term['w'], 0, 0, 4)
        
    # Block 6: A_right (j 2-3) * B_3_bottom (j 2-3, k 4)
    for term in schoolbook_2x2x1_terms:
        add_term(term['u'], term['v'], term['w'], 0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
