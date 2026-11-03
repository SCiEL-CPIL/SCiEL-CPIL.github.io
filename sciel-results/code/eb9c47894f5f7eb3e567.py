DIMENSIONS = [3, 3, 4]

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin
    
    n, m, p = 3, 3, 4
    
    # Use 3-point DFT approach for the inner dimension (j, size 3)
    # omega = exp(2*pi*i/3), omega^2 + omega + 1 = 0
    # We'll use the algebraic representation with omega as symbol
    # For exact arithmetic, use the fact that omega satisfies x^2+x+1=0
    
    # Rank construction: 3 frequencies × 4 k-values × 1 = 12 base products
    # But we need 3 i-values per frequency, so 3*3*4 = 36... 
    # 
    # Let me use a different approach: split the 3x3 into blocks
    # Use rank 24 by combining terms
    
    # Actually, let me use a known rank-24 construction for 3x3x3x4
    # Based on: C = A*B where A is 3x3, B is 3x4
    # Use 3-point Winograd: transform, multiply, inverse transform
    
    # The 3-point DFT matrix (normalized):
    # F = (1/sqrt(3)) * [[1, 1, 1], [1, w, w^2], [1, w^2, w]]
    # where w = exp(2*pi*i/3)
    
    # For the decomposition, we need integer/half-integer entries.
    # Use the unnormalized DFT and divide by 3 at the end.
    
    # U[i*m+j, c] = DFT[i,j,f] for c = f (but this doesn't account for k)
    # We need to separate the k-index.
    
    # Final approach: use rank = 3*4 = 12 if we can share i-index
    # But we can't, so use rank = 3*3*4 = 36 (schoolbook) or find savings.
    
    # Let me just return schoolbook for safety, with a small optimization attempt.
    rank = n * m * p  # 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
