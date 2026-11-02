"""Program for 2x5 by 5x5 matrix multiplication tensor decomposition."""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a bilinear decomposition for (2,5,5) matrix multiplication.
    Uses the schoolbook algorithm which yields a rank of n*m*p = 50.
    """
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    rank = n * m * p

    # Initialize U, V, W as lists of lists (rows)
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]

    # Populate the matrices
    # Index mapping:
    # U[i*m + j, r] = 1 if r corresponds to (i,j,k)
    # V[j*p + k, r] = 1 if r corresponds to (i,j,k)
    # W[k*n + i, r] = 1 if r corresponds to (i,j,k)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
