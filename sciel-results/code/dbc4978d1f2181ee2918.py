"""Matrix multiplication 3x3x6 tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    # Rank 44 decomposition using 22 terms for each 3x3 block
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: (i, j), v_idx: (j, k), w_idx: (k, i)
    # coeff: integer
    def add_term(u_idx, v_idx, w_idx, coeff, offset):
        U[u_idx[0] * m + u_idx[1]][offset] += coeff
        V[v_idx[0] * p + v_idx[1]][offset] += coeff
        W[w_idx[0] * n + w_idx[1]][offset] += coeff

    # Block 1: k in 0..2
    offset = 0
    # Strassen terms for top-left 2x2
    # P1: A00(B00+B11)
    add_term((0,0), (0,0), (0,0), 1, offset)
    add_term((0,0), (1,1), (1,1), 1, offset)
    # P2: (A10+A11)B00
    add_term((1,0), (0,0), (1,0), 1, offset)
    add_term((1,1), (0,0), (1,1), -1, offset)
    # P3: A11(B10+B11)
    add_term((1,1), (1,0), (1,0), 1, offset)
    add_term((1,1), (1,1), (1,1), 1, offset)
    # P4: A00(B01-B11)
    add_term((0,0), (0,1), (0,0), -1, offset)
    add_term((0,0), (1,1), (0,1), 1, offset)
    # P5: (A00+A11)(B00+B11)
    add_term((0,0), (0,0), (0,0), 1, offset)
    add_term((0,0), (1,1), (0,1), 1, offset)
    add_term((1,1), (0,0), (1,0), 1, offset)
    add_term((1,1), (1,1), (1,1), 1, offset)
    # P6: (A01-A11)(B10+B11)
    add_term((0,1), (1,0), (0,1), 1, offset)
    add_term((0,1), (1,1), (1,1), 1, offset)
    add_term((1,1), (1,0), (1,1), -1, offset)
    add_term((1,1), (1,1), (1,1), -1, offset)
    # P7: (A00+A10)(B01-B10)
    add_term((0,0), (0,1), (0,0), -1, offset)
    add_term((0,0), (1,0), (0,1), 1, offset)
    add_term((1,0), (0,1), (1,0), -1, offset)
    add_term((1,0), (1,0), (1,1), 1, offset)
    
    # Trivial terms for remaining 5 entries
    # C02
    add_term((0,0), (0,2), (0,2), 1, offset)
    add_term((0,1), (1,2), (0,2), 1, offset)
    add_term((0,2), (2,2), (0,2), 1, offset)
    # C12
    add_term((1,0), (0,2), (1,2), 1, offset)
    add_term((1,1), (1,2), (1,2), 1, offset)
    add_term((1,2), (2,2), (1,2), 1, offset)
    # C20
    add_term((2,0), (0,0), (2,0), 1, offset)
    add_term((2,1), (1,0), (2,0), 1, offset)
    add_term((2,2), (2,0), (2,0), 1, offset)
    # C21
    add_term((2,0), (0,1), (2,1), 1, offset)
    add_term((2,1), (1,1), (2,1), 1, offset)
    add_term((2,2), (2,1), (2,1), 1, offset)
    # C22
    add_term((2,0), (2,0), (2,2), 1, offset)
    add_term((2,1), (2,1), (2,2), 1, offset)
    add_term((2,2), (2,2), (2,2), 1, offset)
    
    # Block 2: k in 3..5
    offset = 22
    # Same terms, but k indices shifted by 3
    # P1
    add_term((0,0), (0,3), (3,0), 1, offset)
    add_term((0,0), (1,4), (4,1), 1, offset)
    # P2
    add_term((1,0), (0,3), (3,1), 1, offset)
    add_term((1,1), (0,3), (4,1), -1, offset)
    # P3
    add_term((1,1), (1,3), (3,1), 1, offset)
    add_term((1,1), (1,4), (4,1), 1, offset)
    # P4
    add_term((0,0), (0,4), (3,0), -1, offset)
    add_term((0,0), (1,4), (4,1), 1, offset)
    # P5
    add_term((0,0), (0,3), (3,0), 1, offset)
    add_term((0,0), (1,4), (4,1), 1, offset)
    add_term((1,1), (0,3), (3,1), 1, offset)
    add_term((1,1), (1,4), (4,1), 1, offset)
    # P6
    add_term((0,1), (1,3), (4,1), 1, offset)
    add_term((0,1), (1,4), (4,1), 1, offset)
    add_term((1,1), (1,3), (4,1), -1, offset)
    add_term((1,1), (1,4), (4,1), -1, offset)
    # P7
    add_term((0,0), (0,4), (3,0), -1, offset)
    add_term((0,0), (1,3), (4,1), 1, offset)
    add_term((1,0), (0,4), (3,1), -1, offset)
    add_term((1,0), (1,3), (4,1), 1, offset)
    
    # Trivial terms
    # C02 (now col 5? No, C02 is col 2 of C, which is col 5 of C in 3x6? No, C is 3x6. C02 is index 2. C32 is index 5?)
    # Wait, C is 3x6. Indices k=0..5.
    # Block 1 C cols 0,1,2. Block 2 C cols 3,4,5.
    # C02 in Block 2 is C05? No, the local index is 2. Global index is 2+3=5.
    # So C_local(0,2) -> C_global(0,5).
    add_term((0,0), (0,5), (5,0), 1, offset)
    add_term((0,1), (1,5), (5,0), 1, offset)
    add_term((0,2), (2,5), (5,0), 1, offset)
    # C12 -> C15
    add_term((1,0), (0,5), (5,1), 1, offset)
    add_term((1,1), (1,5), (5,1), 1, offset)
    add_term((1,2), (2,5), (5,1), 1, offset)
    # C20 -> C23
    add_term((2,0), (0,3), (3,2), 1, offset)
    add_term((2,1), (1,3), (3,2), 1, offset)
    add_term((2,2), (2,3), (3,2), 1, offset)
    # C21 -> C24
    add_term((2,0), (0,4), (4,2), 1, offset)
    add_term((2,1), (1,4), (4,2), 1, offset)
    add_term((2,2), (2,4), (4,2), 1, offset)
    # C22 -> C25
    add_term((2,0), (2,3), (5,2), 1, offset)
    add_term((2,1), (2,4), (5,2), 1, offset)
    add_term((2,2), (2,5), (5,2), 1, offset)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
