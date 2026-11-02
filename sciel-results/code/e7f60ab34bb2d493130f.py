for i in range(n):
    for k in range(p):
        for j1 in range(m):
            for j2 in range(m):
                s = 0
                for r in range(rank):
                    s += W[k*n+i][r] * U[i*m+j1][r] * V[j2*p+k][r]
                target = 1 if j1 == j2 else 0
