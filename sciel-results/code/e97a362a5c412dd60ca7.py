def evaluate(terms):
    total = np.zeros((n, m, p), dtype=int)
    for u, v, w in terms:
        # u: (n, m)
        # v: (m, p)
        # w: (p, n)
        term = u[:, :, None] * v[None, :, :] * w.T[:, None, :]
        total += term
    return total
