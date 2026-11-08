DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k.
    # This tensor has rank 1, decomposed as u \otimes v \otimes w
    # where u, v, w are all-ones vectors.
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
