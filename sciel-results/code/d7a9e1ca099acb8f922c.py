U[u_idx(0,0)][col] = 1  # A[0,0]
V[v_idx(0,4)][col] = 1  # B[0,4]
W[w_idx(4,0)][col] += 1 # C[4,0] which is C[0,0] of the 5th col block

U[u_idx(0,1)][col] = 1  # A[0,1]
V[v_idx(1,4)][col] = 1  # B[1,4]
W[w_idx(4,0)][col] += 1
