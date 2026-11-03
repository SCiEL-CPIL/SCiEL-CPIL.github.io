"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # Rank 32 decomposition
    # 4 Strassen blocks (2x2x2) of rank 7 each = 28
    # 1 Tail block (2x4x1) of rank 4 = 4
    # Total rank 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)] # 8 x 32
    V = [[0] * rank for _ in range(m * p)] # 20 x 32
    W = [[0] * rank for _ in range(p * n)] # 10 x 32
    
    col = 0

    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # a_base_i, a_base_j: global start index of A block (2x2)
        # b_base_i, b_base_j: global start index of B block (2x2)
        # c_base_i, c_base_j: global start index of C block (2x2)
        
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            # W index is k*n + i.
            # C local (i, j) -> global i = c_base_i + i, k = c_base_j + j
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i

        # M1
        u = {map_a(0): 1, map_a(3): 1}
        v = {map_b(0): 1, map_b(3): 1}
        w = {map_c(0): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        # M2
        u = {map_a(2): 1, map_a(3): 1}
        v = {map_b(0): 1}
        w = {map_c(2): 1, map_c(3): -1}
        add_rank1_term(u, v, w)

        # M3
        u = {map_a(0): 1}
        v = {map_b(1): 1, map_b(3): -1}
        w = {map_c(1): 1, map_c(3): 1}
        add_rank1_term(u, v, w)

        # M4
        u = {map_a(3): 1}
        v = {map_b(2): 1, map_b(0): -1}
        w = {map_c(0): 1, map_c(2): 1}
        add_rank1_term(u, v, w)

        # M5
        u = {map_a(0): 1, map_a(1): 1}
        v = {map_b(3): 1}
        w = {map_c(0): -1, map_c(1): 1}
        add_rank1_term(u, v, w)

        # M6
        u = {map_a(2): 1, map_a(0): -1}
        v = {map_b(0): 1, map_b(1): 1}
        w = {map_c(3): 1}
        add_rank1_term(u, v, w)

        # M7
        u = {map_a(1): 1, map_a(3): -1}
        v = {map_b(2): 1, map_b(3): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)

    def add_tail_2x4_1():
        # Multiplies full A (2x4) by B_col4 (4x1)
        # Results in C_col4 (2x1)
        # Pairs:
        # 1. A_00, A_10 with B_04 -> C_04, C_14
        # 2. A_01, A_11 with B_14 -> C_04, C_14
        # 3. A_02, A_12 with B_24 -> C_04, C_14
        # 4. A_03, A_13 with B_34 -> C_04, C_14
        
        # C_04 index: k=4, i=0 -> 4*2 + 0 = 8
        # C_14 index: k=4, i=1 -> 4*2 + 1 = 9
        
        # Term 1
        u = {0: 1, 4: 1} # A_00, A_10
        v = {4: 1}       # B_04
        w = {8: 1, 9: 1} # C_04, C_14
        add_rank1_term(u, v, w)

        # Term 2
        u = {1: 1, 5: 1} # A_01, A_11
        v = {9: 1}       # B_14
        w = {8: 1, 9: 1} # C_04, C_14
        add_rank1_term(u, v, w)

        # Term 3
        u = {2: 1, 6: 1} # A_02, A_12
        v = {14: 1}      # B_24
        w = {8: 1, 9: 1} # C_04, C_14
        add_rank1_term(u, v, w)

        # Term 4
        u = {3: 1, 7: 1} # A_03, A_13
        v = {19: 1}      # B_34
        w = {8: 1, 9: 1} # C_04, C_14
        add_rank1_term(u, v, w)

    # Block 1: A1 (cols 0,1) * B1 (rows 0,1, cols 0,1)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A1 (cols 0,1) * B1 (rows 0,1, cols 2,3)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # Block 3: A2 (cols 2,3) * B2 (rows 2,3, cols 0,1)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 4: A2 (cols 2,3) * B2 (rows 2,3, cols 2,3)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Tail: A_full * B_col4
    add_tail_2x4_1()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
