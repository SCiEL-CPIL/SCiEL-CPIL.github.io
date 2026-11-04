# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS
    
    def verify_factors(U, V, W, n, m, p):
        R = len(U[0])
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randint(-10, 10, (n, m))
            B = np.random.randint(-10, 10, (m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(R):
                for i in range(n):
                    alpha = 0.0
                    for j in range(m):
                        alpha += U[i*m+j][c] * A[i,j]
                    for k in range(p):
                        beta = 0.0
                        for j in range(m):
                            beta += V[j*p+k][c] * B[j,k]
                        C_computed[i,k] += W[k*n+i][c] * alpha * beta
            if not np.allclose(C_computed, C_expected):
                return False
        return True

    # Try 41
    # 23 factors for 3x3x3
    # I will use a known set.
    # If I don't have it, I will use 45.
    
    # I will use 45.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
