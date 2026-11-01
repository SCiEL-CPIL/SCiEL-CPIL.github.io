  for w_idx, coeff in w_local.items():
      wi = w_idx // 2  # i
      W[k0 * n + wi][col] += coeff
  