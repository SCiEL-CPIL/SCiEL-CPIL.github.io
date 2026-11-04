"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Split m=5 into m1=2 and m2=3.
    # Split p=6 into three blocks of size 2.
    # Block 1 (m1=2, p_block=2): 3 * R(2,2,2) = 3 * 7 = 21
    # Block 2 (m2=3, p_block=2): 3 * R(2,3,2) = 3 * 6 = 18
    # Total Rank = 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {index_in_U: coeff}
        v_coeffs: dict {index_in_V: coeff}
        w_coeffs: dict {index_in_W: coeff}
        """
        nonlocal col
        for idx, c in u_coeffs.items():
            if c != 0:
                U[idx][col] += c
        for idx, c in v_coeffs.items():
            if c != 0:
                V[idx][col] += c
        for idx, c in w_coeffs.items():
            if c != 0:
                W[idx][col] += c
        col += 1

    # Helper to get indices
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Strassen terms for 2x2x2
    # Local indices i in 0,1; j in 0,1; k in 0,1
    def strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # M1
        add_term(
            {u_idx(i_off, j_off): 1, u_idx(i_off+1, j_off+1): 1},
            {v_idx(j_off, k_off): 1, v_idx(j_off+1, k_off+1): 1},
            {w_idx(k_off, i_off): 1, w_idx(k_off+1, i_off+1): 1}
        )
        # M2
        add_term(
            {u_idx(i_off+1, j_off): 1, u_idx(i_off+1, j_off+1): 1},
            {v_idx(j_off, k_off): 1},
            {w_idx(k_off, i_off+1): 1, w_idx(k_off+1, i_off+1): -1}
        )
        # M3
        add_term(
            {u_idx(i_off, j_off): 1},
            {v_idx(j_off, k_off+1): 1, v_idx(j_off+1, k_off+1): -1},
            {w_idx(k_off+1, i_off): 1, w_idx(k_off+1, i_off+1): 1}
        )
        # M4
        add_term(
            {u_idx(i_off+1, j_off+1): 1},
            {v_idx(j_off+1, k_off): 1, v_idx(j_off, k_off): -1},
            {w_idx(k_off, i_off): 1, w_idx(k_off, i_off+1): 1}
        )
        # M5
        add_term(
            {u_idx(i_off, j_off): 1, u_idx(i_off, j_off+1): 1},
            {v_idx(j_off+1, k_off+1): 1},
            {w_idx(k_off, i_off): -1, w_idx(k_off+1, i_off): 1}
        )
        # M6
        add_term(
            {u_idx(i_off+1, j_off): 1, u_idx(i_off, j_off): -1},
            {v_idx(j_off, k_off): 1, v_idx(j_off, k_off+1): 1},
            {w_idx(k_off+1, i_off+1): 1}
        )
        # M7
        add_term(
            {u_idx(i_off, j_off+1): 1, u_idx(i_off+1, j_off+1): -1},
            {v_idx(j_off+1, k_off): 1, v_idx(j_off+1, k_off+1): 1},
            {w_idx(k_off, i_off): 1}
        )

    # Schoolbook terms for 2x3x2
    # Local indices i in 0,1; j in 0,1,2; k in 0,1
    def schoolbook_2x3x2(i_off, j_off, k_off):
        for i in range(2):
            for k in range(2):
                for j in range(3):
                    add_term(
                        {u_idx(i_off + i, j_off + j): 1},
                        {v_idx(j_off + j, k_off + k): 1},
                        {w_idx(k_off + k, i_off + i): 1}
                    )

    # Block 1: A1 (2x2) * B1 (2x6)
    # B1 is split into three 2x2 blocks: columns 0-1, 2-3, 4-5
    # A1 is rows 0-1, cols 0-1 of A
    # j_off = 0
    strassen_2x2x2(0, 0, 0)
    strassen_2x2x2(0, 0, 2)
    strassen_2x2x2(0, 0, 4)

    # Block 2: A2 (2x3) * B2 (3x6)
    # B2 is split into three 3x2 blocks: columns 0-1, 2-3, 4-5
    # A2 is rows 0-1, cols 2-4 of A
    # j_off = 2
    schoolbook_2x3x2(0, 2, 0)
    schoolbook_2x3x2(0, 2, 2)
    schoolbook_2x3x2(0, 2, 4)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
