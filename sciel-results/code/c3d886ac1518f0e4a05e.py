        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    result[i, j, k] += u[i] * v[j] * w[k]
