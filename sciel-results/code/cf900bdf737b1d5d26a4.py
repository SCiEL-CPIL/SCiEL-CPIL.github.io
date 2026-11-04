   for i in range(n):
       for j in range(m):
           for k in range(p):
               U[i * m + j, col] = 1
               V[j * p + k, col] = 1
               W[k * n + i, col] = 1
               col += 1
   