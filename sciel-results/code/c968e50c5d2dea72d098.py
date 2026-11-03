      for i in range(n):
          for j in range(m):
              for k in range(p):
                  u_idx = i * m + j
                  v_idx = j * p + k
                  w_idx = k * n + i
                  U[u_idx, col] = 1
                  V[v_idx, col] = 1
                  W[w_idx, col] = 1
      