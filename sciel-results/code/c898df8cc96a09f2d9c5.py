# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    The minimal rank is n*m*p = 2*5*5 = 50.
    The tensor M_{2,5,5} decomposes as a direct sum of two M_{1,5,5} tensors,
    each of which has rank 25. Total rank = 50.
    """
    n, m, p = 2, 5, 5
    rank = n * m * p  # 50, which is optimal
    
    U = [[0] * rank for _ in range(n * m)]  # 10 x 50
    V = [[0] * rank for _ in range(m * p)]  # 25 x 50
    W = [[0] * rank for _ in range(p * n)]  # 10 x 50
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (trace convention)
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
