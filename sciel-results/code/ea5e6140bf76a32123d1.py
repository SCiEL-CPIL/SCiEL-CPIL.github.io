DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 30 decomposition
    # U: (n*m, 30) all 1s
    # V: (m*p, 30) Identity
    # W: (p*n, 30) all 1s
    r = m * p
    U = [[1] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[1] * r for _ in range(p * n)]
    
    for i in range(r):
        V[i][i] = 1
        
    return {"U": U, "V": V, "W": W}
