    for k in range(P):
        b2k = 2 * P + k
        U[a02, col] += 1; V[b2k, col] += 1; W[2 * k, col] += 1
        col += 1
        U[a12, col] += 1; V[b2k, col] += 1; W[2 * k + 1, col] += 1
        col += 1
