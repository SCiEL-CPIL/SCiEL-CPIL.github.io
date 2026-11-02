for k in range(p):
    b2k = 2 * p + k
    U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
    col += 1
    U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
    col += 1
