W[k_off * n + i_off][col] = 1      # W[0*n+0] = +1 ← WRONG (should be -1)
W[(k_off+1) * n + i_off][col] = -1  # W[1*n+0] = -1 ← WRONG (should be 0)
