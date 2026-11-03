i_idx = np.repeat(np.arange(n), m * p)
j_idx = np.tile(np.repeat(np.arange(m), p), n)
k_idx = np.tile(np.arange(p), n * m)
