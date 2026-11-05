for i in range(n):
    for j in range(m):
        for k in range(p):
            expected = 1.0
            actual = 0.0
            for r in range(rank):
                actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            if abs(expected - actual) > tol:
                return False
