"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen 2x2x2 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen instance
    # a_idx: indices in U for A_block (top-left, top-right, bot-left, bot-right)
    # b_idx: indices in V for B_block (top-left, top-right, bot-left, bot-right)
    # w_idx: indices in W for C_block (top-left, top-right, bot-left, bot-right)
    # col_offset: starting column in U, V, W
    def add_strassen(a_idx, b_idx, w_idx, col_offset):
        a11, a12, a21, a22 = a_idx
        b11, b12, b21, b22 = b_idx
        c11, c12, c21, c22 = w_idx
        
        # M1 = (a11+a22)(b11+b22) -> c11, c22
        U[a11][col_offset] += 1
        U[a22][col_offset] += 1
        V[b11][col_offset] += 1
        V[b22][col_offset] += 1
        W[c11][col_offset] += 1
        W[c22][col_offset] += 1
        
        # M2 = (a21+a22)b11 -> c21, c22
        U[a21][col_offset + 1] += 1
        U[a22][col_offset + 1] += 1
        V[b11][col_offset + 1] += 1
        W[c21][col_offset + 1] += 1
        W[c22][col_offset + 1] += 1
        
        # M3 = a11(b12-b22) -> c12, c22
        U[a11][col_offset + 2] += 1
        V[b12][col_offset + 2] += 1
        V[b22][col_offset + 2] -= 1
        W[c12][col_offset + 2] += 1
        W[c22][col_offset + 2] += 1
        
        # M4 = a22(b21-b11) -> c11, c21
        U[a22][col_offset + 3] += 1
        V[b21][col_offset + 3] += 1
        V[b11][col_offset + 3] -= 1
        W[c11][col_offset + 3] += 1
        W[c21][col_offset + 3] += 1
        
        # M5 = (a11+a12)b22 -> c11(-), c12
        U[a11][col_offset + 4] += 1
        U[a12][col_offset + 4] += 1
        V[b22][col_offset + 4] += 1
        W[c11][col_offset + 4] -= 1
        W[c12][col_offset + 4] += 1
        
        # M6 = (a21-a11)(b11+b12) -> c22
        U[a21][col_offset + 5] += 1
        U[a11][col_offset + 5] -= 1
        V[b11][col_offset + 5] += 1
        V[b12][col_offset + 5] += 1
        W[c22][col_offset + 5] += 1
        
        # M7 = (a12-a22)(b21+b22) -> c11
        U[a12][col_offset + 6] += 1
        U[a22][col_offset + 6] -= 1
        V[b21][col_offset + 6] += 1
        V[b22][col_offset + 6] += 1
        W[c11][col_offset + 6] += 1

    # Instance 0: A1 * B1 -> C1
    # A1: U[0], U[1], U[4], U[5]
    # B1: V[0], V[1], V[4], V[5]
    # C1: W[0], W[2], W[1], W[3]
    add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3], 0)
    
    # Instance 1: A2 * B3 -> C1
    # A2: U[2], U[3], U[6], U[7]
    # B3: V[8], V[9], V[12], V[13]
    # C1: W[0], W[2], W[1], W[3]
    add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3], 7)
    
    # Instance 2: A1 * B2 -> C2
    # A1: U[0], U[1], U[4], U[5]
    # B2: V[2], V[3], V[6], V[7]
    # C2: W[4], W[6], W[5], W[7]
    add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7], 14)
    
    # Instance 3: A2 * B4 -> C2
    # A2: U[2], U[3], U[6], U[7]
    # B4: V[10], V[11], V[14], V[15]
    # C2: W[4], W[6], W[5], W[7]
    add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7], 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
