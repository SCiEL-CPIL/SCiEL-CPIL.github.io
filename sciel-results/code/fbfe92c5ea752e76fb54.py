def slice_method(n, m, p):
    rank = m
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for j in range(m):
        for i in range(n):
            U[i*m + j, j] = 1.0
        for k in range(p):
            V[j*p + k, j] = 1.0
        for k in range(p):
            for i in range(n):
                W[k*n + i, j] = 1.0
    return U, V, W, rank
