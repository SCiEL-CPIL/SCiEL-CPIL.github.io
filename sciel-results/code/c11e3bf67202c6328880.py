for i in range(n):
    for j in range(m):
        for k in range(p):
            target = 1.0
            calc = 0.0
            for r in range(rank):
                calc += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            if abs(calc - target) > 1e-9:
                pass
