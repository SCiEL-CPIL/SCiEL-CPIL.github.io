def construct():
    n, m, p = 2, 3, 9
    rank = 1
    U = [[1] for _ in range(n * m)]  # shape (6, 1)
    V = [[1] for _ in range(m * p)]  # shape (27, 1)
    W = [[1] for _ in range(p * n)]  # shape (18, 1)
    return {"U": U, "V": V, "W": W}
