"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]

    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            for i in range(4):
                U[i][c] = U_forms[t][i]
            if t == 0:
                V[2*u][c] += 1
                V[p + 2*u + 1][c] += 1
            elif t == 1:
                V[2*u][c] += 1
            elif t == 2:
                V[2*u + 1][c] += 1
                V[p + 2*u + 1][c] -= 1
            elif t == 3:
                V[p + 2*u][c] += 1
                V[2*u][c] -= 1
            elif t == 4:
                V[p + 2*u + 1][c] += 1
            elif t == 5:
                V[2*u][c] += 1
                V[2*u + 1][c] += 1
            elif t == 6:
                V[p + 2*u][c] += 1
                V[p + 2*u + 1][c] += 1

            if t == 0:
                W[2*u][c] += 1
                W[n*(2*u+1)+1][c] += 1
            elif t == 1:
                W[n*(2*u)+1][c] += 1
                W[n*(2*u+1)+1][c] -= 1
            elif t == 2:
                W[n*(2*u+1)][c] += 1
                W[n*(2*u+1)+1][c] += 1
            elif t == 3:
                W[2*u][c] += 1
                W[n*(2*u)+1][c] += 1
            elif t == 4:
                W[2*u][c] -= 1
                W[n*(2*u+1)][c] += 1
            elif t == 5:
                W[n*(2*u+1)+1][c] += 1
            elif t == 6:
                W[2*u][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
