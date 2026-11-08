import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (8, rank)
    # V shape: (20, rank)
    # W shape: (10, rank)
    
    # Rank 30 decomposition
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def add_term(u_indices, v_indices, w_indices, sign=1):
        nonlocal col
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        
        for idx in u_indices:
            if idx is not None:
                U[idx][col] += sign
        for idx in v_indices:
            if idx is not None:
                V[idx][col] += sign
        for idx in w_indices:
            if idx is not None:
                W[idx][col] += sign
                
        col += 1

    def strassen_2x2x2(a_11, a_12, a_21, a_22, b_11, b_12, b_21, b_22, c_11, c_12, c_21, c_22):
        # M1: (A11 + A22) * (B11 + B22)
        add_term([a_11, a_22], [b_11, b_22], [c_11, c_22])
        # M2: (A21 + A22) * B11
        add_term([a_21, a_22], [b_11], [c_21, c_22], -1)
        # Note: W signs are +1 for c_21, -1 for c_22.
        # My add_term applies same sign to all.
        # I need to handle signs per index or split calls.
        # Let's split calls for different signs.
        
        # Re-doing M2
        # (A21 + A22) * B11 -> C21 += M, C22 -= M
        add_term([a_21, a_22], [b_11], [c_21])
        add_term([a_21, a_22], [b_11], [c_22], -1)
        
        # M3: A11 * (B12 - B22)
        add_term([a_11], [b_12, b_22], [c_12, c_22])
        
        # M4: A22 * (B21 - B11)
        add_term([a_22], [b_21, b_11], [c_11, c_21])
        
        # M5: (A11 + A12) * B22
        add_term([a_11, a_12], [b_22], [c_11], -1)
        add_term([a_11, a_12], [b_22], [c_12])
        
        # M6: (A21 - A11) * (B11 + B12)
        add_term([a_21, a_11], [b_11, b_12], [c_22])
        
        # M7: (A12 - A22) * (B21 + B22)
        add_term([a_12, a_22], [b_21, b_22], [c_11])

    def standard_2x2x2(a_11, a_12, a_21, a_22, b_11, b_12, b_21, b_22, c_11, c_12, c_21, c_22):
        add_term([a_11], [b_11], [c_11])
        add_term([a_12], [b_21], [c_11])
        add_term([a_11], [b_12], [c_12])
        add_term([a_12], [b_22], [c_12])
        add_term([a_21], [b_11], [c_21])
        add_term([a_22], [b_21], [c_21])
        add_term([a_21], [b_12], [c_22])
        add_term([a_22], [b_22], [c_22])

    def standard_2x2x1(a_11, a_12, a_21, a_22, b_1, b_2, c_1, c_2):
        add_term([a_11], [b_1], [c_1])
        add_term([a_12], [b_2], [c_1])
        add_term([a_21], [b_1], [c_2])
        add_term([a_22], [b_2], [c_2])

    def block_2x2x5(a_11, a_12, a_21, a_22, b_11, b_12, b_13, b_14, b_15, b_21, b_22, b_23, b_24, b_25, c_11, c_12, c_13, c_14, c_15, c_21, c_22, c_23, c_24, c_25):
        # B cols 0,1 -> Strassen
        strassen_2x2x2(a_11, a_12, a_21, a_22, b_11, b_12, b_21, b_22, c_11, c_12, c_21, c_22)
        
        # B col 2 -> Standard 2x2x1
        # b_13 is B[0,2], b_23 is B[1,2]
        standard_2x2x1(a_11, a_12, a_21, a_22, b_13, b_23, c_13, c_23)
        
        # B cols 3,4 -> Standard 2x2x2
        standard_2x2x2(a_11, a_12, a_21, a_22, b_14, b_15, b_24, b_25, c_14, c_15, c_24, c_25)

    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    a1_11 = u_idx(0, 0); a1_12 = u_idx(0, 1); a1_21 = u_idx(1, 0); a1_22 = u_idx(1, 1)
    b1_11 = v_idx(0, 0); b1_12 = v_idx(0, 1); b1_13 = v_idx(0, 2); b1_14 = v_idx(0, 3); b1_15 = v_idx(0, 4)
    b1_21 = v_idx(1, 0); b1_22 = v_idx(1, 1); b1_23 = v_idx(1, 2); b1_24 = v_idx(1, 3); b1_25 = v_idx(1, 4)
    c1_11 = w_idx(0, 0); c1_12 = w_idx(0, 1); c1_13 = w_idx(0, 2); c1_14 = w_idx(0, 3); c1_15 = w_idx(0, 4)
    c1_21 = w_idx(1, 0); c1_22 = w_idx(1, 1); c1_23 = w_idx(1, 2); c1_24 = w_idx(1, 3); c1_25 = w_idx(1, 4)
    
    block_2x2x5(a1_11, a1_12, a1_21, a1_22, b1_11, b1_12, b1_13, b1_14, b1_15, b1_21, b1_22, b1_23, b1_24, b1_25, c1_11, c1_12, c1_13, c1_14, c1_15, c1_21, c1_22, c1_23, c1_24, c1_25)
    
    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    a2_11 = u_idx(0, 2); a2_12 = u_idx(0, 3); a2_21 = u_idx(1, 2); a2_22 = u_idx(1, 3)
    b2_11 = v_idx(2, 0); b2_12 = v_idx(2, 1); b2_13 = v_idx(2, 2); b2_14 = v_idx(2, 3); b2_15 = v_idx(2, 4)
    b2_21 = v_idx(3, 0); b2_22 = v_idx(3, 1); b2_23 = v_idx(3, 2); b2_24 = v_idx(3, 3); b2_25 = v_idx(3, 4)
    c2_11 = w_idx(0, 0); c2_12 = w_idx(0, 1); c2_13 = w_idx(0, 2); c2_14 = w_idx(0, 3); c2_15 = w_idx(0, 4)
    c2_21 = w_idx(1, 0); c2_22 = w_idx(1, 1); c2_23 = w_idx(1, 2); c2_24 = w_idx(1, 3); c2_25 = w_idx(1, 4)
    
    block_2x2x5(a2_11, a2_12, a2_21, a2_22, b2_11, b2_12, b2_13, b2_14, b2_15, b2_21, b2_22, b2_23, b2_24, b2_25, c2_11, c2_12, c2_13, c2_14, c2_15, c2_21, c2_22, c2_23, c2_24, c2_25)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
