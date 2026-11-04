<<<<<<< SEARCH
def _build_rank28():
    """
    Build a rank-28 decomposition for 3x4x4 matmul using Strassen structure.
    
    Decomposition: j = 2*j1 + j2, k = 2*k1 + k2 where j1,k1,j2,k2 in {0,1}
    For each (j1,k1) block, use 7 Strassen terms. Total rank = 4*7 = 28.
    
    T[i, 2*j1+j2, 2*k1+k2] = sum_t U[i,2*j1+j2,t] * V[2*j1+j2,2*k1+k2,t] * W[2*k1+k2,i,t]
    
    For block (j1,k1) with Strassen term t:
    U[i, 2*j1+j2, t] = U2[j2, t]  (independent of i)
    V[2*j1+j2, 2*k1+k2, t] = V2[k2, t]  (depends on k2 only within this block)
    W[2*k1+k2, i, t] = W2[j2, t]  (depends on j2 only)
    
    Then sum_t U2[j2,t]*V2[k2,t]*W2[j2,t] -- this is not 1 in general.
    
    The issue is that Strassen has U depending on (i,j) combined, not separately.
    In 2x2x2, U[i,t] where i is the row index of A (= row of C in output).
    But in our 3x4x4, U[i,j,t] has both i and j.
    
    Let me try: U[i, 2*j1+j2, t] = U2[j2, t] * delta(i, something)
    This is getting too complicated. Let me just do a smarter random search.
    """
    n, m, p = 3, 4, 4
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W
=======
def _build_rank44():
    """
    Build a rank-44 decomposition for 3x4x4 matmul.
    - i in {0,1}: use 4 Strassen 2x2x2 blocks (4*7 = 28 terms)
    - i = 2: use schoolbook (4*4 = 16 terms)
    Total: 44 terms
    """
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    U2, V2, W2 = _strassen_2x2()
    
    col = 0
    # Strassen blocks for i in {0,1}
    for jb in range(2):  # j-block: 0 or 1
        for kb in range(2):  # k-block: 0 or 1
            for t in range(7):
                # U[i*m + (2*jb+jb_inner), t] = U2[i, t]
                for i in range(2):
                    for jb_inner in range(2):
                        j = 2*jb + jb_inner
                        U[i*m + j][col] = U2[i][t]
                # V[(2*jb+jb_inner)*p + (2*kb+kb_inner), t] = V2[jb_inner, t]
                for jb_inner in range(2):
                    for kb_inner in range(2):
                        j = 2*jb + jb_inner
                        k = 2*kb + kb_inner
                        V[j*p + k][col] = V2[jb_inner][t]
                # W[(2*kb+kb_inner)*n + i, t] = W2[kb_inner, t]
                for i in range(2):
                    for kb_inner in range(2):
                        k = 2*kb + kb_inner
                        W[k*n + i][col] = W2[kb_inner][t]
                col += 1
    
    # Schoolbook for i = 2
    for j in range(m):
        for k in range(p):
            U[2*m + j][col] = 1.0
            V[j*p + k][col] = 1.0
            W[k*n + 2][col] = 1.0
            col += 1
    
    return U, V, W
>>>>>>> REPLACE
Now update the `construct()` function to use this rank-44 decomposition:

