"""Direct-sum decomposition of 2x3x10 matrix multiplication into two 2x3x5 subproblems."""
DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    half = p // 2
    rank_per_block = n * m * half
    R = 2 * rank_per_block

    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    for block_idx in range(2):
        offset = block_idx * rank_per_block
        for i_idx in range(n):
            for j_idx in range(m):
                for k_idx in range(half):
                    r = offset + (i_idx * m + j_idx) * half + k_idx
                    U[i_idx * m + j_idx][r] = 1
                    V[j_idx * p + block_idx * half + k_idx][r] = 1
                    W[(block_idx * half + k_idx) * n + i_idx][r] = 1

    return {"U": U, "V": V, "W": W}
