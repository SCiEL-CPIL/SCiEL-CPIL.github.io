"""Rewritten program for 2x5x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6

    # We decompose the 2x5x6 problem into:
    #   Part 1: 2x2x6 (A cols 0,1; B rows 0,1)
    #   Part 2: 2x3x6 (A cols 2,3,4; B rows 2,3,4)
    #
    # For 2x2x6: use 3 Strassen 2x2x2 blocks (one per pair of k-values) -> rank 21
    # For 2x3x6: split into 2x2x6 (cols 2,3) + 2x1x6 (col 4)
    #   2x2x6: rank 21 (3 Strassen blocks)
    #   2x1x6: rank 12 (naive)
    #   Total for 2x3x6: 33
    # Grand total: 21 + 33 = 54
    #
    # However, we can do better by using a more efficient decomposition for 2x2x6.
    # Known: R(2x2x2) = 7 (Strassen). For 2x2x4, R = 11 is known.
    # For 2x2x6, we can split k into 4+2: R(2x2x4) + R(2x2x2) = 11 + 7 = 18.
    #
    # Let me implement the 11-rank 2x2x4 decomposition.
    # The 11-rank decomposition for 2x2x4 is a known result.
    # I'll construct it carefully.

    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * rank for _ in range(p * n)]  # 12 x 54

    def strassen_2x2x2(U, V, W, j0, k0, col):
        """Add Strassen decomposition for 2x2x2 block."""
        # P1 = (A_0,j0 + A_1,j0+1)(B_j0,k0 + B_j0+1,k0+1)
        U[0*m + j0][col] += 1
        U[1*m + j0+1][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + 0][col] += 1
        W[(k0+1)*n + 1][col] += 1
        col += 1

        # P2 = (A_1,j0 + A_1,j0+1) B_j0,k0
        U[1*m + j0][col] += 1
        U[1*m + j0+1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + 1][col] += 1
        W[(k0+1)*n + 1][col] -= 1
        col += 1

        # P3 = A_0,j0 (B_j0,k0+1 - B_j0+1,k0+1)
        U[0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + 0][col] += 1
        W[(k0+1)*n + 1][col] += 1
        col += 1

        # P4 = A_1,j0+1 (B_j0+1,k0 - B_j0,k0)
        U[1*m + j0+1][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + 0][col] += 1
        W[k0*n + 1][col] += 1
        col += 1

        # P5 = (A_0,j0 + A_0,j0+1) B_j0+1,k0+1
        U[0*m + j0][col] += 1
        U[0*m + j0+1][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + 0][col] -= 1
        W[(k0+1)*n + 0][col] += 1
        col += 1

        # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k0+1)
        U[1*m + j0][col] += 1
        U[0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + 1][col] += 1
        col += 1

        # P7 = (A_0,j0+1 - A_1,j0+1)(B_j0+1,k0 + B_j0+1,k0+1)
        U[0*m + j0+1][col] += 1
        U[1*m + j0+1][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + 0][col] += 1
        col += 7
        return col

    def naive_2x1x6(U, V, W, j, col):
        """Add naive decomposition for 2x1x6 block at A col j; B row j; all k."""
        for k in range(p):
            U[0*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
        return col

    # Use the straightforward approach: 3 Strassen blocks for each 2x2x6, naive for 2x1x6
    col = 0
    # Part 1: 2x2x6 for j=0,1
    for k0 in [0, 2, 4]:
        col = strassen_2x2x2(U, V, W, 0, k0, col)

    # Part 2: 2x3x6 for j=2,3,4
    # Sub-part 2a: 2x2x6 for j=2,3
    for k0 in [0, 2, 4]:
        col = strassen_2x2x2(U, V, W, 2, k0, col)

    # Sub-part 2b: 2x1x6 for j=4
    col = naive_2x1x6(U, V, W, 4, col)

    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
