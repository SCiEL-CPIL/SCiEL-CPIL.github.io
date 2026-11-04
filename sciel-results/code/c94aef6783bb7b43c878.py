import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # Rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col_idx = 0
    
    # Strassen factors for 2x2x2
    # Indices in A_L (2x2):
    # x00: A[0,0] -> U row 0
    # x01: A[0,1] -> U row 1
    # x10: A[1,0] -> U row 3
    # x11: A[1,1] -> U row 4
    # Indices in B_L (2x2) for block b:
    # y00: B[0, 2b] -> V row 2b
    # y01: B[0, 2b+1] -> V row 2b+1
    # y10: B[1, 2b] -> V row 8+2b
    # y11: B[1, 2b+1] -> V row 8+2b+1
    
    # W indices for block b:
    # Z00 (C[0, 2b]): W row (2b)*2 + 0 = 4b
    # Z01 (C[0, 2b+1]): W row (2b+1)*2 + 0 = 4b+2
    # Z10 (C[1, 2b]): W row (2b)*2 + 1 = 4b+1
    # Z11 (C[1, 2b+1]): W row (2b+1)*2 + 1 = 4b+3

    for b in range(4):
        # --- Strassen Part (7 cols) ---
        # M1 = (x00 + x11)(y00 + y11)
        c = col_idx
        U[0][c] = 1; U[4][c] = 1
        V[2*b][c] = 1; V[8+2*b+1][c] = 1
        W[4*b][c] = 1; W[4*b+3][c] = 1
        col_idx += 1

        # M2 = (x10 + x11)y00
        c = col_idx
        U[3][c] = 1; U[4][c] = 1
        V[2*b][c] = 1
        W[4*b+1][c] = 1; W[4*b+3][c] = -1
        col_idx += 1

        # M3 = x00(y01 - y11)
        c = col_idx
        U[0][c] = 1
        V[2*b+1][c] = 1; V[8+2*b+1][c] = -1
        W[4*b+2][c] = 1; W[4*b+3][c] = 1
        col_idx += 1

        # M4 = x11(y10 - y00)
        c = col_idx
        U[4][c] = 1
        V[8+2*b][c] = 1; V[2*b][c] = -1
        W[4*b][c] = 1; W[4*b+1][c] = 1
        col_idx += 1

        # M5 = (x00 + x01)y11
        c = col_idx
        U[0][c] = 1; U[1][c] = 1
        V[8+2*b+1][c] = 1
        W[4*b][c] = -1; W[4*b+2][c] = 1
        col_idx += 1

        # M6 = (x10 - x00)(y00 + y01)
        c = col_idx
        U[3][c] = 1; U[0][c] = -1
        V[2*b][c] = 1; V[2*b+1][c] = 1
        W[4*b+3][c] = 1
        col_idx += 1

        # M7 = (x01 - x11)(y10 + y11)
        c = col_idx
        U[1][c] = 1; U[4][c] = -1
        V[8+2*b][c] = 1; V[8+2*b+1][c] = 1
        W[4*b][c] = 1; W[4*b+2][c] = 1
        col_idx += 1

        # --- Direct Part (4 cols) for A[:, 2] * B[2, :] ---
        # Q00 = A[0, 2] * B[2, 2b]
        c = col_idx
        U[2][c] = 1 # A[0, 2] is row 0, col 2 -> index 2
        V[16+2*b][c] = 1 # B[2, 2b] is row 2, col 2b -> index 16+2b
        W[4*b][c] = 1 # C[0, 2b]
        col_idx += 1

        # Q01 = A[0, 2] * B[2, 2b+1]
        c = col_idx
        U[2][c] = 1
        V[16+2*b+1][c] = 1
        W[4*b+2][c] = 1 # C[0, 2b+1]
        col_idx += 1

        # Q10 = A[1, 2] * B[2, 2b]
        c = col_idx
        U[5][c] = 1 # A[1, 2] is row 1, col 2 -> index 5
        V[16+2*b][c] = 1
        W[4*b+1][c] = 1 # C[1, 2b]
        col_idx += 1

        # Q11 = A[1, 2] * B[2, 2b+1]
        c = col_idx
        U[5][c] = 1
        V[16+2*b+1][c] = 1
        W[4*b+3][c] = 1 # C[1, 2b+1]
        col_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
