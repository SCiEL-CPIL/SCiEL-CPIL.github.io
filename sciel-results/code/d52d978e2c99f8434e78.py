        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - expected) > 1e-6:
                        return False
