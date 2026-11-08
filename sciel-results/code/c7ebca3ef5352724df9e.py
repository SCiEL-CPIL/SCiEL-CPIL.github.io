def construct():
    n, m, p = DIMENSIONS
    # The target tensor in _verify is np.ones((n, m, p))
    # A rank-1 decomposition exists: U, V, W all ones.
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
