"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # C = A @ B, A is 2x2, B is 2x12.
    # Split B into two halves: B0 = B[:, 0:6], B1 = B[:, 6:12].
    # C0 = A @ B0, C1 = A @ B1, each 2x2 @ 2x6.
    # For each 2x2 @ 2x6, split the 6 columns into three 2x2 blocks.
    # Use Strassen (rank 7) for each 2x2 @ 2x2.
    # Total rank = 2 halves * 3 blocks * 7 = 42.

    n, m, p = DIMENSIONS
    rank = 42

    U = [[0] * rank for _ in range(n * m)]   # (4, 42)
    V = [[0] * rank for _ in range(m * p)]   # (24, 42)
    W = [[0] * rank for _ in range(p * n)]   # (24, 42)

    # Strassen: Z = X * Y, X,Y are 2x2.
    # 7 products:
    # P0 = (x00+x11)*(y00+y11)
    # P1 = (x10+x11)*y00
    # P2 = x00*(y10-y11)
    # P3 = x11*(y01-y00)
    # P4 = (x00+x01)*y11
    # P5 = (x10-x00)*(y00+y10)
    # P6 = (x11-x01)*(y01+y11)
    # z00 = P0 + P3 - P4 + P6
    # z01 = P2 + P4
    # z10 = P1 + P3
    # z11 = P0 - P1 + P2 + P5

    # X factors (for A): x_ij -> U row i*m+j
    X = [
        {0: 1, 3: 1},          # P0: x00, x11
        {2: 1, 3: 1},          # P1: x10, x11
        {0: 1},                # P2: x00
        {3: 1},                # P3: x11
        {0: 1, 1: 1},          # P4: x00, x01
        {2: 1, 0: -1},         # P5: x10, -x00
        {3: 1, 1: -1},         # P6: x11, -x01
    ]

    # Y factors (for each 2x2 block Y of B): y_jt -> V row j*p + (base_k+t)
    Y = [
        {(0, 0): 1, (1, 1): 1},   # P0: y00, y11
        {(0, 0): 1},               # P1: y00
        {(1, 0): 1, (1, 1): -1},  # P2: y10, -y11
        {(0, 1): 1, (0, 0): -1},  # P3: y01, -y00
        {(1, 1): 1},               # P4: y11
        {(0, 0): 1, (1, 0): 1},   # P5: y00, y10
        {(0, 1): 1, (1, 1): 1},   # P6: y01, y11
    ]

    # Z factors (output): z_it -> W row (base_k+t)*n + i
    Z = [
        {(0, 0): 1, (1, 1): 1},   # P0
        {(1, 0): 1, (1, 1): -1},  # P1
        {(0, 1): 1, (1, 1): 1},   # P2
        {(0, 0): 1, (1, 0): 1},   # P3
        {(0, 0): -1, (0, 1): 1},  # P4
        {(1, 1): 1},              # P5
        {(0, 0): 1},              # P6
    ]

    col = 0
    for half in range(2):
        for sub in range(3):
            base_k = half * 6 + sub * 2
            for r in range(7):
                for (xrow, coeff) in X[r].items():
                    U[xrow][col] += coeff
                for ((j, t), coeff) in Y[r].items():
                    vrow = j * p + base_k + t
                    V[vrow][col] += coeff
                for ((i, t), coeff) in Z[r].items():
                    wrow = (base_k + t) * n + i
                    W[wrow][col] += coeff
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
