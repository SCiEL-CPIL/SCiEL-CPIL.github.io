        for c in range(min(rank, n*m*p)):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            if i < n and j < m and k < p:
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
   