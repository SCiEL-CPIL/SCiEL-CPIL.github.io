   # 4x1x4 part: A_R (4x1) * B_RL (1x4) -> C (4x4)
   # Rank 16
   for i in range(4):
       for k in range(4):
           U[i*5 + 4][col] = 1 # j=4
           V[4*5 + k][col] = 1 # j=4, k in 0..3
           W[k*4 + i][col] = 1
           col += 1
   