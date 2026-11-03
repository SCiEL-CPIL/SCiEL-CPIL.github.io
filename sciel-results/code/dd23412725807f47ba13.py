for i in range(n):
    for k in range(p):
        for j in range(m):
            s = 0.0
            for r in range(rank):
                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_decomposed[i, k] += s
