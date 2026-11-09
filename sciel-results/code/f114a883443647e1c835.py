    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
       