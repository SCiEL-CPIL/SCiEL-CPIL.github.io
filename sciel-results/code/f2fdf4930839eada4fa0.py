for i in range(n):
    for j in range(m):
        for k in range(p):
            T_computed[i,j,k] = U[i,j] * V[j,k] * W[k,i]
