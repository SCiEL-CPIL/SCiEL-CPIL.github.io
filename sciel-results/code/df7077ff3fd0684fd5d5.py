    C_computed = np.zeros((n, p))
    for r in range(rank):
        M_u = np.dot(U[:, r], A_test.flatten())
        M_v = np.dot(V[:, r], B_test.flatten())
        M_r = M_u * M_v
        C_computed += M_r * W[:, r].reshape(p, n).T
