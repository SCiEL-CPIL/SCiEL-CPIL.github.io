n_blocks = BASE // m
reshaped = uncovered.reshape(n_blocks, m)
gains = reshaped.sum(axis=0)  # gains[r] = number of uncovered in residue class r
best_r = np.argmax(gains)
