def construct():
    n, m, p = 2, 3, 10
    rank = 3 * p + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For 2x3xp, use the known rank 3p+4 decomposition
    # Base terms (4 terms)
    # Term 0: A[0,0] * B[0,k] for all k -> contributes to C[0,k]
    # Term 1: A[1,0] * B[0,k] for all k -> contributes to C[1,k]
    # Term 2: A[0,1] * B[1,k] for all k -> contributes to C[0,k]
    # Term 3: A[1,1] * B[1,k] for all k -> contributes to C[1,k]
    # Then for columns 2..p-1, we need additional terms
    
    # Actually, this is getting complicated. Let me use a different approach.
    
    # Use the fact that 2x3 matrix mult by 3xp can be done in 3p+4
    # by sharing the first two columns' computation
    
    pass
