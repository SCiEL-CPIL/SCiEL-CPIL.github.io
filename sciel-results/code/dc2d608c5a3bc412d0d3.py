"""Tensor decomposition for 2x5x6 matrix multiplication, rank 52."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    rank = 52

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    def _add(u, v, w):
        nonlocal col
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1

    # --- Block 0: i=0, j=0 --- (10 cols)
    for k in range(p):
        _add(0, k, k)
    for k in range(p):
        _add(1, k, k)
    _add(1, 5, 6)
    _add(5, 0, 12)

    # --- Block 1: i=0, j=1 --- (10 cols)
    for k in range(p):
        _add(5, k, k)
    for k in range(p):
        _add(1, k, k)
    _add(1, 5, 6)
    _add(10, 0, 12)

    # --- Block 2: i=0, j=2 --- (10 cols)
    for k in range(p):
        _add(10, k, k)
    for k in range(p):
        _add(1, k, k)
    _add(1, 5, 6)
    _add(15, 0, 12)

    # --- Block 3: i=0, j=3 --- (10 cols)
    for k in range(p):
        _add(15, k, k)
    for k in range(p):
        _add(1, k, k)
    _add(1, 5, 6)
    _add(20, 0, 12)

    # --- Block 4: i=0, j=4 --- (10 cols)
    for k in range(p):
        _add(20, k, k)
    for k in range(p):
        _add(1, k, k)
    _add(1, 5, 6)
    _add(25, 0, 12)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
