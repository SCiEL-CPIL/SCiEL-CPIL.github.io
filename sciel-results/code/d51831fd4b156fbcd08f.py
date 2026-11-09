for i in range(n):
    for j in range(m):
        for k in range(p):
            val = 0.0
            for c in range(r):
                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            if abs(val - 1.0) > 1e-8:
                return False
