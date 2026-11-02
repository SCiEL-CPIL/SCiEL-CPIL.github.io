import itertools

def construct():
    n, m, p = 3, 3, 8
    # Schoolbook rank is n*m*p = 72
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Generate all (i, j, k) tuples
    # To make it distinct, we can iterate in a different order
    # e.g., k-major order instead of i-major
    indices = list(itertools.product(range(p), range(n), range(m)))
    # Shuffle or just use this order
    # Let's use k, i, j order
    for col, (k, i, j) in enumerate(indices):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        
    return {"U": U, "V": V, "W": W}
