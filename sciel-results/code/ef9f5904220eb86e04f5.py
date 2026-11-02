"""Program for 4x4 by 4x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    # We split the 8 columns of B into two groups of 4.
    # This gives us two independent 4x4 by 4x4 multiplications.
    # Each 4x4 by 4x4 multiplication has rank 16 (schoolbook is optimal for 4x4).
    # Total rank = 2 * 16 = 32.
    rank = 2 * n * n  # 32
    U = [[0] * rank for _ in range(n * m)]  # 16 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 32 x 32

    # For each of the two column blocks (k_offset = 0 and 4)
    for k_block in range(2):
        base = k_block * n * n  # 0 and 16
        for i in range(n):
            for j in range(m):
                for k_local in range(n):
                    c = base + i * m + j
                    k = k_local + k_block * n
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
