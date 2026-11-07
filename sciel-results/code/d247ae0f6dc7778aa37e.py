def _lhs(n, dim, rng):
    pts = np.zeros((n, dim))
    for d in range(dim):
        perm = rng.permutation(n)
        pts[:, d] = (perm + rng.random(n)) / n
    return pts
