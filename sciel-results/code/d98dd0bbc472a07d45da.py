def _build_rank_20():
    n, m, p = DIMENSIONS
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # n=2, m=2, p=10
    # U indices: 0..3 (i*m+j)
    # V indices: 0..19 (j*p+k)
    # W indices: 0..19 (k*n+i)
    
    for b in range(5):
        # Block b corresponds to column b of B and C
        # V indices for B: j=0 -> b, j=1 -> 10+b
        # W indices for C: k=b -> 2b, 2b+1 (i=0, 1)
        
        # Term 1: A[0,0] * B[0,b] -> C[0,b]
        # U index 0, V index b, W index 2b
        r = b * 4
        U[0][r] = 1
        V[b][r] = 1
        W[2*b][r] = 1
        
        # Term 2: A[0,1] * B[1,b] -> C[0,b]
        # U index 1, V index 10+b, W index 2b
        r = b * 4 + 1
        U[1][r] = 1
        V[10+b][r] = 1
        W[2*b][r] = 1
        
        # Term 3: A[1,0] * B[0,b] -> C[1,b]
        # U index 2, V index b, W index 2b+1
        r = b * 4 + 2
        U[2][r] = 1
        V[b][r] = 1
        W[2*b+1][r] = 1
        
        # Term 4: A[1,1] * B[1,b] -> C[1,b]
        # U index 3, V index 10+b, W index 2b+1
        r = b * 4 + 3
        U[3][r] = 1
        V[10+b][r] = 1
        W[2*b+1][r] = 1
        
    return {"U": U, "V": V, "W": W}
