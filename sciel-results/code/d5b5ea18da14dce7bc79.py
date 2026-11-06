    for i, j in net:
        col_i = all_vectors[:, i]
        col_j = all_vectors[:, j]
        # min goes to i, max goes to j
        all_vectors[:, i] = np.minimum(col_i, col_j)
        all_vectors[:, j] = np.maximum(col_i, col_j)
