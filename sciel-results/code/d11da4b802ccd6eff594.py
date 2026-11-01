   val = sum(U[row_u][c] * V[row_v][c] * W[row_w][c] for c in range(rank))
   i = row_u // m
   j = row_u % m
   j_v = row_v // p
   k = row_v % p
   k_w = row_w // n
   i_w = row_w % n
   is_valid = (j == j_v and i == i_w)
   