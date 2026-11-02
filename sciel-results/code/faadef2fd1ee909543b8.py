DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-60 decomposition: 3 rows * 20 terms per 5x5 mat-vec product
    # For each row i, we compute C[i,:] = A[i,:] * B
    # Using 20 bilinear terms per row (saving 5 from schoolbook 25)
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 15 x 60
    V = [[0] * rank for _ in range(m * p)]  # 25 x 60
    W = [[0] * rank for _ in range(p * n)]  # 15 x 60
    
    col = 0
    for i in range(n):
        # For row i, compute 5 output entries C[i,0..4]
        # A[i,0..4] * B[0..4, 0..4]
        # Use 20 terms: 5 diagonal + 15 for off-diagonal (Strassen-like savings)
        
        # Diagonal terms: A[i,j] * B[j,k] where k==j gives C[i,j]
        # But we need all C[i,k], not just diagonal
        
        # Let's use a different grouping.
        # Group the 25 products into 20 bilinear forms.
        
        # Terms 1-5: A[i,j] * B[j,j] for j=0..4 (contributes to C[i,j])
        for j in range(5):
            U[i * m + j][col] = 1
            V[j * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1
        
        # Terms 6-15: For pairs (j,k) with j<k, use Strassen-like
        # For each pair j<k, we need A[i,j]*B[j,k] and A[i,k]*B[k,j]
        # Instead of 2 mults, use: 
        #   t1 = (A[i,j]+A[i,k])*(B[j,k]+B[k,j])
        #   t2 = A[i,j]*B[j,k]  -- wait, that's still 2
        # 
        # Strassen for 2x2: to compute [[a,b],[c,d]]*[[e,f],[g,h]]
        # m1=(a+b)(e+f), m2=(c+d)(e+f), m3=a(g-h), m4=d(f-e), m5=(a+c)e, m6=(b+d)h, m7=(a-c)(h-g)
        # 
        # For our case, we have a row vector times a matrix.
        # Let me just use 15 off-diagonal terms directly (no savings) 
        # and accept rank 25 per row = 75 total. That's not an improvement.
        
        # Let me try a different approach: use rank-7 Strassen on 2x2 blocks.
        # Partition the 5x5 B into overlapping 2x2 blocks.
        # This is complex. Let me just use schoolbook for now but verify correctness.
        pass
    
    # Fallback: use schoolbook rank 75
    rank = n * m * p
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
# EVOLVE-BLOCK-END
