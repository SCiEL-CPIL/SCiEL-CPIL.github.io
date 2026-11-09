"""Matrix multiplication 3x4x5 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = 3, 4, 5
    rank = m * p  # 20
    
    U = [[0] * rank for _ in range(n * m)]  # shape (12, 20)
    V = [[0] * rank for _ in range(m * p)]  # shape (20, 20)
    W = [[0] * rank for _ in range(p * n)]  # shape (15, 20)
    
    for t in range(rank):
        j = t % m    # column of A / row of B
        k = t // m   # row of B / column of C
        
        # U[i*m + j, t] = 1 for all i
        for i in range(n):
            U[i * m + j][t] = 1
        
        # V[j*p + k, t] = 1
        V[j * p + k][t] = 1
        
        # W[k*n + i, t] = 1 for all i
        for i in range(n):
            W[k * n + i][t] = 1
    
    return {"U": U, "V": V, "W": W}
