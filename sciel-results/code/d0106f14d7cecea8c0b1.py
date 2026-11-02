def construct():
    n, m, p = 4, 4, 8
    # Rank 1 decomposition of the all-ones tensor
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
