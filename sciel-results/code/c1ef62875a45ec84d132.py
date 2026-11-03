DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 1 decomposition
    rank = 1
    # U shape (n*m, rank)
    U = [[1] * rank for _ in range(n * m)]
    # V shape (m*p, rank)
    V = [[1] * rank for _ in range(m * p)]
    # W shape (p*n, rank)
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
