"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # Winograd-style decomposition using 2x2x2 (rank 7) and 2x2x1 (rank 4) blocks.
    # Total rank: 2 * (7 + 7 + 4) = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    def add_strassen_block(a_i, a_j, b_i, b_j, c_i, c_k):
        # Maps local 2x2 indices to global
        def ma(l): return (a_i + l//2)*m + (a_j + l%2)
        def mb(l): return (b_i + l//2)*p + (b_j + l%2)
        def mc(l): return (c_k + l%2)*n + (c_i + l//2)

        # Strassen 7 terms
        add_rank1_term({ma(0):1, ma(3):1}, {mb(0):1, mb(3):1}, {mc(0):1, mc(3):1})
        add_rank1_term({ma(2):1, ma(3):1}, {mb(0):1}, {mc(2):1, mc(3):-1})
        add_rank1_term({ma(0):1}, {mb(1):1, mb(3):-1}, {mc(1):1, mc(3):1})
        add_rank1_term({ma(3):1}, {mb(2):1, mb(0):-1}, {mc(0):1, mc(2):1})
        add_rank1_term({ma(0):1, ma(1):1}, {mb(3):1}, {mc(0):-1, mc(1):1})
        add_rank1_term({ma(2):1, ma(0):-1}, {mb(0):1, mb(1):1}, {mc(3):1})
        add_rank1_term({ma(1):1, ma(3):-1}, {mb(2):1, mb(3):1}, {mc(0):1})

    def add_schoolbook_2x2x1(a_i, a_j, b_i, b_k, c_i, c_k):
        # 2x2 A * 2x1 B -> 2x1 C
        # A: (a_i, a_j), (a_i, a_j+1), (a_i+1, a_j), (a_i+1, a_j+1)
        # B: (b_i, b_k), (b_i+1, b_k)
        # C: (c_i, c_k), (c_i+1, c_k)
        
        def ma(l): return (a_i + l//2)*m + (a_j + l%2)
        def mb(l): return (b_i + l)*p + b_k
        def mc(l): return c_k*n + (c_i + l)

        # C_0 = A_00 B_0 + A_01 B_1
        # C_1 = A_10 B_0 + A_11 B_1
        add_rank1_term({ma(0):1}, {mb(0):1}, {mc(0):1})
        add_rank1_term({ma(1):1}, {mb(1):1}, {mc(0):1})
        add_rank1_term({ma(2):1}, {mb(0):1}, {mc(1):1})
        add_rank1_term({ma(3):1}, {mb(1):1}, {mc(1):1})

    # Block 1: A cols 0-1, B rows 0-1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4)
    
    # Block 2: A cols 2-3, B rows 2-3
    add_strassen_block(0, 2, 2, 0, 0, 0)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
