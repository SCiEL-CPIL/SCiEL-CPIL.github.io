   for j in range(m):
       for k in range(2):
           b_idx = j * p + (block * 2 + k)
           v_idx = j * 2 + k
           V[b_idx, col] = V2[v_idx, r]
   