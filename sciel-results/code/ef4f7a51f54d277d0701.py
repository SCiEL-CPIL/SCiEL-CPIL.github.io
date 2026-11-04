"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    
    # Verify a rank-19 decomposition for 3x3 matrix multiplication.
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    # Must satisfy: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    
    # Use the full schoolbook decomposition (rank 27) as it is guaranteed correct.
    # A true rank-19 decomposition requires non-trivial linear combinations that
    # cannot be easily hardcoded without verification.
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*m+j] = 1
                v[j*p+k] = 1
                w[k*n+i] = 1
                terms.append((u, v, w))
    
    # Verify the decomposition
    def verify(terms, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for (u, v, w) in terms:
                        total += u[i*m+j] * v[j*p+k] * w[k*n+i]
                    if total != 1:
                        return False
        return True
    
    if verify(terms, n, m, p):
        rank = len(terms)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for r, (u, v, w) in enumerate(terms):
            for idx in range(9):
                U[idx][r] = u[idx]
                V[idx][r] = v[idx]
                W[idx][r] = w[idx]
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: should not happen with schoolbook
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
