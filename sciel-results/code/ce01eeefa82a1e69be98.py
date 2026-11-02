"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    
    Strategy: Use Strassen's algorithm (rank 7) for the 2x2 * 2x2 sub-problem
    (first two columns of B), and schoolbook (rank 4) for the remaining column.
    Total rank = 7 + 4 = 11.
    
    Matrix multiplication: C = A * B where A is 2x2, B is 2x3, C is 2x3.
    
    Indexing conventions:
    - U: shape (n*m, rank) = (4, rank). Row index = i*m + j where i is row of A, j is col of A.
    - V: shape (m*p, rank) = (6, rank). Row index = j*p + k where j is row of B, k is col of B.
    - W: shape (p*n, rank) = (6, rank). Row index = k*n + i where k is row of C, i is col of C.
    
    The decomposition satisfies: for each (i, k), sum over j of A[i,j]*B[j,k] = sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    More precisely, the tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if the product A[i,j]*B[j,k] contributes to C[i,k], which it always does, so T[i,j,k] = 1 for all valid indices... wait, no.
    
    The tensor for matrix multiplication M_{n,m,p} has entries T[i, j, k] = 1 for all i, j, k, because C[i,k] = sum_j A[i,j] B[j,k]. So the tensor is all-ones. The rank is the minimum number of rank-1 tensors that sum to the all-ones tensor in the appropriate sense.
    
    Actually, the tensor T is defined by T[i, j, k] = 1 for all i in [n], j in [m], k in [p]. A decomposition of rank r gives matrices U (n*m x r), V (m*p x r), W (p*n x r) such that for all i, j, k:
    T[i, j, k] = sum_{r=0}^{r-1} U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    Since T[i, j, k] = 1 for all i, j, k, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    For Strassen on 2x2 * 2x2:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B.
    C[0,0] = ae + bg
    C[0,1] = af + bh
    C[1,0] = ce + dg
    C[1,1] = cf + dh
    
    Strassen factors (7 multiplications):
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C[0,0] = m1 - m4 + m5 - m7
    C[0,1] = m3 + m5
    C[1,0] = m2 + m4
    C[1,1] = m1 - m3 + m2 - m6
    
    Now let's map to our indexing:
    A indices: i in {0,1}, j in {0,1}. U row = i*2 + j.
    B indices: j in {0,1}, k in {0,1} for first two columns. V row = j*3 + k.
    C indices: k in {0,1} for first two columns, i in {0,1}. W row = k*2 + i.
    
    For the first 7 factors (Strassen on B[:, :2]):
    
    Factor 0: m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    U: A[0,0]=1, A[1,1]=1, others 0. Row indices: 0 (0*2+0), 3 (1*2+1).
    U = [1, 0, 0, 1]
    V: B[0,0]=1, B[1,1]=1. Row indices: 0 (0*3+0), 4 (1*3+1).
    V = [1, 0, 0, 1, 0, 0]
    W: C[0,0] gets +1, C[1,1] gets +1. Row indices: 0 (0*2+0), 3 (1*2+1).
    W = [1, 0, 0, 1, 0, 0]
    
    Factor 1: m2 = (A[1,0] + A[1,1]) * B[0,0]
    U: A[1,0]=1, A[1,1]=1. Row indices: 2 (1*2+0), 3 (1*2+1).
    U = [0, 0, 1, 1]
    V: B[0,0]=1. Row index: 0.
    V = [1, 0, 0, 0, 0, 0]
    W: C[1,0] gets +1. Row index: 2 (1*2+0).
    W = [0, 0, 0, 0, 1, 0]
    
    Factor 2: m3 = A[0,0] * (B[0,1] - B[1,1])
    U: A[0,0]=1. Row index: 0.
    U = [1, 0, 0, 0]
    V: B[0,1]=1, B[1,1]=-1. Row indices: 1 (0*3+1), 4 (1*3+1).
    V = [0, 1, 0, 0, -1, 0]
    W: C[0,1] gets +1, C[1,1] gets -1. Row indices: 1 (0*2+1), 3 (1*2+1).
    W = [0, 1, 0, -1, 0, 0]
    
    Factor 3: m4 = A[1,1] * (B[1,0] - B[0,0])
    U: A[1,1]=1. Row index: 3.
    U = [0, 0, 0, 1]
    V: B[1,0]=1, B[0,0]=-1. Row indices: 3 (1*3+0), 0 (0*3+0).
    V = [-1, 0, 0, 1, 0, 0]
    W: C[0,0] gets -1, C[1,0] gets +1. Row indices: 0, 2.
    W = [-1, 0, 1, 0, 0, 0]
    
    Factor 4: m5 = (A[0,0] + A[0,1]) * B[1,1]
    U: A[0,0]=1, A[0,1]=1. Row indices: 0, 1.
    U = [1, 1, 0, 0]
    V: B[1,1]=1. Row index: 4.
    V = [0, 0, 0, 0, 1, 0]
    W: C[0,0] gets +1, C[0,1] gets +1. Row indices: 0, 1.
    W = [1, 1, 0, 0, 0, 0]
    
    Factor 5: m6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U: A[1,0]=1, A[0,0]=-1. Row indices: 2, 0.
    U = [-1, 0, 1, 0]
    V: B[0,0]=1, B[0,1]=1. Row indices: 0, 1.
    V = [1, 1, 0, 0, 0, 0]
    W: C[1,1] gets -1. Row index: 3.
    W = [0, 0, 0, -1, 0, 0]
    
    Factor 6: m7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U: A[0,1]=1, A[1,1]=-1. Row indices: 1, 3.
    U = [0, 1, 0, -1]
    V: B[1,0]=1, B[1,1]=1. Row indices: 3, 4.
    V = [0, 0, 0, 1, 1, 0]
    W: C[0,0] gets -1, C[1,1] gets -1. Row indices: 0, 3.
    W = [-1, 0, 0, -1, 0, 0]
    
    Now for the last column (k=2) of B, we use schoolbook:
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Factor 7: A[0,0] * B[0,2] -> C[0,2]
    U: A[0,0]=1. Row index: 0.
    U = [1, 0, 0, 0]
    V: B[0,2]=1. Row index: 2 (0*3+2).
    V = [0, 0, 1, 0, 0, 0]
    W: C[0,2]=1. Row index: 4 (2*2+0).
    W = [0, 0, 0, 0, 1, 0]
    
    Factor 8: A[0,1] * B[1,2] -> C[0,2]
    U: A[0,1]=1. Row index: 1.
    U = [0, 1, 0, 0]
    V: B[1,2]=1. Row index: 5 (1*3+2).
    V = [0, 0, 0, 0, 0, 1]
    W: C[0,2]=1. Row index: 4.
    W = [0, 0, 0, 0, 1, 0]
    
    Factor 9: A[1,0] * B[0,2] -> C[1,2]
    U: A[1,0]=1. Row index: 2.
    U = [0, 0, 1, 0]
    V: B[0,2]=1. Row index: 2.
    V = [0, 0, 1, 0, 0, 0]
    W: C[1,2]=1. Row index: 5 (2*2+1).
    W = [0, 0, 0, 0, 0, 1]
    
    Factor 10: A[1,1] * B[1,2] -> C[1,2]
    U: A[1,1]=1. Row index: 3.
    U = [0, 0, 0, 1]
    V: B[1,2]=1. Row index: 5.
    V = [0, 0, 0, 0, 0, 1]
    W: C[1,2]=1. Row index: 5.
    W = [0, 0, 0, 0, 0, 1]
    
    Let me verify factor 0: 
    For i=0, j=0, k=0: U[0,0]=1, V[0,0]=1, W[0,0]=1. Product = 1.
    For i=0, j=0, k=1: U[0,0]=1, V[0,1]=0, W[0,1]=0. Product = 0.
    For i=0, j=1, k=0: U[0,1]=0, ... = 0.
    For i=1, j=1, k=1: U[0,3]=1, V[0,4]=1, W[0,3]=1. Product = 1.
    For i=1, j=1, k=0: U[0,3]=1, V[0,3]=0, W[0,2]=0. Product = 0.
    For i=1, j=0, k=1: U[0,2]=0, ... = 0.
    For i=0, j=1, k=1: U[0,1]=0, ... = 0.
    For i=1, j=0, k=0: U[0,2]=0, ... = 0.
    
    So factor 0 contributes to (i=0,j=0,k=0) and (i=1,j=1,k=1). That's correct for (a+d)(e+h).
    
    I'll implement this and verify computationally.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    rank = 11
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Strassen factors for first two columns (k=0,1)
    # Factor 0: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Factor 1: (A[1,0]+A[1,1]) * B[0,0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Factor 2: A[0,0] * (B[0,1]-B[1,1])
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    # Factor 3: A[1,1] * (B[1,0]-B[0,0])
    U[3][3] = 1
    V[0][3] = -1; V[3][3] = 1
    W[0][3] = -1; W[2][3] = 1
    
    # Factor 4: (A[0,0]+A[0,1]) * B[1,1]
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # Factor 5: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = -1
    
    # Factor 6: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = -1; W[3][6] = -1
    
    # Schoolbook factors for third column (k=2)
    # Factor 7: A[0,0] * B[0,2]
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Factor 8: A[0,1] * B[1,2]
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Factor 9: A[1,0] * B[0,2]
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Factor 10: A[1,1] * B[1,2]
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
