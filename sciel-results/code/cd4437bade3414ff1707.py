"""Program for 2x2x14 matrix multiplication tensor, rank-49 via Strassen blocks."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 14
    # Split the p=14 columns into (p//2)=7 blocks of 2 columns each.
    # Each block is a 2x2 x 2x2 multiplication, decomposed by Strassen (rank 7).
    # Total rank = (p//2) * 7 = 49.
    blocks = p // 2  # 7
    assert blocks * 2 == p
    rank = blocks * 7
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49

    # Strassen 2x2 coefficients (0-indexed local indices).
    # A-coefficients: dict (i,j) -> value, per product s
    Acoef = [
        {(0,0): 1, (1,1): 1},              # m1
        {(1,0): 1, (1,1): 1},              # m2
        {(0,0): 1},                        # m3
        {(1,1): 1},                        # m4
        {(0,0): 1, (0,1): 1},              # m5
        {(1,0): 1, (0,0): -1},             # m6
        {(0,1): 1, (1,1): -1},             # m7
    ]
    # B-coefficients: dict (j, k_local) -> value, per product s
    Bcoef = [
        {(0,0): 1, (1,1): 1},              # m1
        {(0,0): 1},                        # m2
        {(0,1): 1, (1,1): -1},             # m3
        {(1,0): 1, (0,0): -1},             # m4
        {(1,1): 1},                        # m5
        {(0,0): 1, (0,1): 1},              # m6
        {(1,0): 1, (1,1): 1},              # m7
    ]
    # C-coefficients: dict (k_local, i) -> value, per product s
    Ccoef = [
        {(0,0): 1, (1,1): 1},              # m1
        {(0,1): 1, (1,1): -1},             # m2
        {(1,0): 1, (1,1): 1},              # m3
        {(0,0): 1, (0,1): 1},              # m4
        {(0,0): -1, (1,0): 1},             # m5
        {(1,1): 1},                        # m6
        {(0,0): 1},                        # m7
    ]

    for b in range(blocks):
        for s in range(7):
            c = b * 7 + s
            # U
            for (i, j), val in Acoef[s].items():
                U[i * m + j][c] += val
            # V
            for (j, kl), val in Bcoef[s].items():
                k = 2 * b + kl
                V[j * p + k][c] += val
            # W
            for (kl, i), val in Ccoef[s].items():
                k = 2 * b + kl
                W[k * n + i][c] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
