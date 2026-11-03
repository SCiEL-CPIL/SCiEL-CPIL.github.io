     def _validate(U, V, W, n, m, p):
         ...
         for i in range(n):
             for j in range(m):
                 for k in range(p):
                     s = 0
                     ...
                     for c in range(rank):
                         s += U[ui][c] * V[vj][c] * W[wk][c]
                     if abs(s - 1.0) > 1e-9:
                         return False
         return True
     