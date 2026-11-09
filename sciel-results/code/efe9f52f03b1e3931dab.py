import time

def construct():
    t_start = time.time()
    deadline = t_start + 115

    n, m, p = 2, 4, 5

    # Strassen decomposition for 2x2x2
    strassen_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    # Base rank: 4 Strassen blocks (28) + 8 schoolbook for last column = 36
    base_rank = 4 * 7 + 8
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]

    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li * m + j0 + lj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0 + lj) * p + k0 + lk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0 + lk) * n + li][col] = strassen_W[r][lk * 2 + li]
                col += 1

    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # Try to reduce rank by eliminating linear dependencies
    # Build the tensor and check for linear dependencies in the rank-1 terms
    # Each rank-1 term is a vector in R^(n*m) x R^(m*p) x R^(p*n)
    # We can try to find if any term is a linear combination of others

    # For now, return the base decomposition
    # A more sophisticated approach would use linear algebra to find
    # linear dependencies and eliminate redundant terms

    return {"U": U, "V": V, "W": W}
