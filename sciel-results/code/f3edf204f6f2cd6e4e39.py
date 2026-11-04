def construct():
    n, m, p = DIMENSIONS
    # The matmul tensor T[i,j,k] = 1 for all valid (i,j,k)
    # This is the outer product of three all-ones vectors, so rank = 1
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": W, "W": W}
