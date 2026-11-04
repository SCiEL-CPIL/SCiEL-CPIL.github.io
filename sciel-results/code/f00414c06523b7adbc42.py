"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition: split A (3x5) into A1 (1x5) and A2 (2x5)
    # Split B (5x5) into B1 (5x2) and B2 (5x3)
    # C = A*B where C is 3x5
    # C11 = A1*B1 (1x2), C12 = A1*B2 (1x3)
    # C21 = A2*B1 (2x2), C22 = A2*B2 (2x3)
    #
    # Use Strassen for 2x2x2 blocks, schoolbook for others
    # Total rank: 10 + 15 + 7 + 12 = 44 (using schoolbook for 2x2x3 = 12)
    # Actually let's be more careful:
    # A1*B1: 1x5 * 5x2 = 10 terms (schoolbook)
    # A1*B2: 1x5 * 5x3 = 15 terms (schoolbook)
    # A2*B1: 2x5 * 5x2 = use Strassen on 2x2x2 after splitting B1 into 5x1 and 5x1
    #   A2 = [A2a(2x2), A2b(2x2), A2c(2x1)]
    #   B1 = [B1a(5x1), B1b(5x1)]
    #   This is getting complex. Let me just use schoolbook for all: 10+15+10+15 = 50
    #
    # Better approach: use the full 3x5x5 with some Strassen-like savings
    # Split A into 3 blocks of 1x5, B into 5 blocks of 5x1
    # That's just schoolbook = 75
    #
    # Let's try a different split:
    # A (3x5) -> split rows: A_top (1x5), A_bot (2x5)
    # B (5x5) -> split cols: B_left (5x2), B_right (5x3)
    #
    # C_top_left (1x2) = A_top * B_left: rank 10 (schoolbook)
    # C_top_right (1x3) = A_top * B_right: rank 15 (schoolbook)
    # C_bot_left (2x2) = A_bot * B_left: rank 7 (Strassen)
    # C_bot_right (2x3) = A_bot * B_right: rank 12 (schoolbook)
    # Total: 10 + 15 + 7 + 12 = 44
    #
    # For C_bot_right (2x3): A_bot is 2x5, B_right is 5x3
    # Split A_bot into A_bl (2x2) and A_br (2x3)
    # Split B_right into B_rl (5x1) and B_rr (5x2)
    # C_bl*rl = 2x1: rank 4 (schoolbook)
    # C_bl*rr = 2x2: rank 7 (Strassen)
    # C_br*rl = 3x1: rank 6 (schoolbook)
    # C_br*rr = 3x2: rank 12 (schoolbook)
    # Total for 2x3: 4+7+6+12 = 29... that's worse than 12
    #
    # Just use schoolbook for 2x3: rank 12
    # Total: 10 + 15 + 7 + 12 = 44
    #
    # Let's implement this.
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We'll build U, V, W with total rank = 44
    # Each term is indexed by (i, j, k) for the schoolbook parts,
    # and by Strassen indices for the 2x2x2 part
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 15 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 25 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 15 x 44
    
    col = 0
    
    # Block 1: C_top_left = A_top * B_left (1x2)
    # A_top = row 0 of A (indices 0-4 in A)
    # B_left = cols 0-1 of B
    # For j in 0..4, k in 0..1:
    #   U[0*m + j] = U[j] gets 1
    #   V[j*p + k] = V[5*j + k] gets 1
    #   W[k*n + 0] = W[k*3] gets 1
    for j in range(5):
        for k in range(2):
            U[j][col] = 1.0
            V[j * 5 + k][col] = 1.0
            W[k * 3 + 0][col] = 1.0
            col += 1
    # col = 10
    
    # Block 2: C_top_right = A_top * B_right (1x3)
    # B_right = cols 2-4 of B
    for j in range(5):
        for k in range(2, 5):
            U[j][col] = 1.0
            V[j * 5 + k][col] = 1.0
            W[k * 3 + 0][col] = 1.0
            col += 1
    # col = 25
    
    # Block 3: C_bot_left = A_bot * B_left (2x2) using Strassen
    # A_bot = rows 1-2 of A (indices 5-9 in A, i.e., i=1,2; j=0..4)
    # B_left = cols 0-1 of B
    # Split A_bot into 2x2 blocks:
    #   A11 = A[1,0..1] (2x2), A12 = A[1,2..4] (2x3)
    #   A21 = A[2,0..1] (2x2), A22 = A[2,2..4] (2x3)
    # Wait, A_bot is 2x5. Split into A_bl (2x2) and A_br (2x3).
    # B_left is 5x2. Split into B_tl (2x2) and B_bl (3x2)? No.
    # B_left has 5 rows, 2 cols. Split rows into 2 and 3.
    # B_tl = B[0..1, 0..1] (2x2), B_bl = B[2..4, 0..1] (3x2)
    #
    # C_bot_left = A_bot * B_left = [A_bl, A_br] * [B_tl; B_bl]
    # = A_bl * B_tl + A_br * B_bl
    # A_bl * B_tl is 2x2: use Strassen (rank 7)
    # A_br * B_bl is 2x2: A_br is 2x3, B_bl is 3x2. Rank 12 (schoolbook)
    # Total for block 3: 7 + 12 = 19
    #
    # Hmm, that gives 10 + 15 + 19 + 12 = 56.
    #
    # Let me reconsider. Maybe just use schoolbook for everything: 75.
    # Or try a different split.
    #
    # Actually, let's just do the straightforward schoolbook. It's safe and valid.
    # The directive says to improve, but if I can't find a valid improvement,
    # I should at least return a valid answer.
    
    # Reset
    rank = n * m * p  # 75
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
