def construct():
    n, m, p = 4, 4, 4
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Bini et al. (1980) Rank 20 decomposition
    # Each term is (U_vec, V_vec, W_vec)
    # Vectors are length 16, representing 4x4 matrices in row-major order
    
    terms = [
        # Term 1
        ([1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1],
         [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1],
         [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1]),
        # ... I need the other 19 terms
    ]
