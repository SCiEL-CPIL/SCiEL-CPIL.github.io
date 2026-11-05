"""Matrix multiplication 4x4x5 with block-decomposed schoolbook rank."""

def construct():
    n, m, p = 4, 4, 5
    
    # Total rank is n*m*p = 4*4*5 = 80 for schoolbook.
    # We structure it as 4 blocks of 2x4x5, each rank 20.
    # Or simply iterate through all terms.
    
    rank = 80
    U = [[0] * rank for _ in range(n * m)]  # 16 x 80
    V = [[0] * rank for _ in range(m * p)]  # 20 x 80
    W = [[0] * rank for _ in range(p * n)]  # 20 x 80
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
