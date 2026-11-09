   for i in range(n):
       for j in range(m):
           for k in range(p):
               s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
               if abs(s - 1) > 1e-10:
                   # Error
   