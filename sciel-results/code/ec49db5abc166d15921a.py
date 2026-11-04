"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140

    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_4x4(U, V, W, a_base, b_base, c_base, col):
        for dr in range(2):
            for dc in range(2):
                U[a_base + dr * 4 + dc][col] = 1
                U[a_base + 2 + dr * 4 + 2 + dc][col] = 1
                V[b_base + dr * 9 + dc][col] = 1
                V[b_base + 2 + dr * 9 + 2 + dc][col] = 1
                W[c_base + dc * 4 + dr][col] = 1
                W[c_base + 2 + dc * 4 + 2 + dr][col] = 1
                W[c_base + 2 + dc * 4 + dr][col] = -1
                W[c_base + dc * 4 + 2 + dr][col] = 1
        col += 1

        for dr in range(2):
            for dc in range(2):
                U[a_base + 2 + dr * 4 + dc][col] = 1
                U[a_base + 2 + dr * 4 + 2 + dc][col] = 1
                V[b_base + dr * 9 + dc][col] = 1
                W[c_base + 2 + dc * 4 + dr][col] = 1
                W[c_base + 2 + dc * 4 + 2 + dr][col] = -1
        col += 1

        for dr in range(2):
            for dc in range(2):
                U[a_base + dr * 4 + dc][col] = 1
                V[b_base + 2 + dr * 9 + dc][col] = 1
                V[b_base + 2 + dr * 9 + 2 + dc][col] = -1
                W[c_base + dc * 4 + 2 + dr][col] = 1
                W[c_base + 2 + dc * 4 + 2 + dr][col] = 1
        col += 1

        for dr in range(2):
            for dc in range(2):
                U[a_base + 2 + dr * 4 + 2 + dc][col] = 1
                V[b_base + dr * 9 + 2 + dc][col] = 1
                V[b_base + dr * 9 + dc][col] = -1
                W[c_base + dc * 4 + dr][col] = 1
                W[c_base + 2 + dc * 4 + dr][col] = 1
        col += 1

        for dr in range(2):
            for dc in range(2):
                U[a_base + dr * 4 + dc][col] = 1
                U[a_base + dr * 4 + 2 + dc][col] = 1
                V[b_base + 2 + dr * 9 + 2 + dc][col] = 1
                W[c_base + dc * 4 + dr][col] = -1
                W[c_base + dc * 4 + 2 + dr][col] = 1
        col += 1

        for dr in range(2):
            for dc in range(2):
                U[a_base + 2 + dr * 4 + dc][col] = 1
                U[a_base + dr * 4 + dc][col] = -1
                V[b_base + dr * 9 + dc][col] = 1
                V[b_base + 2 + dr * 9 + dc][col] = 1
                W[c_base + 2 + dc * 4 + 2 + dr][col] = 1
        col += 1

        for dr in range(2):
            for dc in range(2):
                U[a_base + dr * 4 + 2 + dc][col] = 1
                U[a_base + 2 + dr * 4 + 2 + dc][col] = -1
                V[b_base + 2 + dr * 9 + 2 + dc][col] = 1
                V[b_base + dr * 9 + 2 + dc][col] = 1
                W[c_base + dc * 4 + dr][col] = 1
                W[c_base + dc * 4 + 2 + dr][col] = -1
        col += 1

        return col

    def add_schoolbook_4x4_x_4x1(U, V, W, a_base, b_base, c_base, col):
        for i in range(4):
            for j in range(4):
                U[a_base + i * 4 + j][col] = 1
                V[b_base + j * 9][col] = 1
                W[c_base + i][col] = 1
                col += 1
        return col

    col = add_strassen_4x4(U, V, W, 0, 0, 0, col)
    col = add_strassen_4x4(U, V, W, 0, 4, 4, col)
    col = add_schoolbook_4x4_x_4x1(U, V, W, 0, 8, 8, col)

    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
