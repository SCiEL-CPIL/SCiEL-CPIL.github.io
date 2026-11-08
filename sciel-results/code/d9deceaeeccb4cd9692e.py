  # V is (N_V, rank), W is (N_W, rank)
  # We want M (N_V*N_W, rank)
  # M[b*N_W + c, r] = V[b, r] * W[c, r]
  # This is equivalent to:
  # M = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(N_V * N_W, rank)
  