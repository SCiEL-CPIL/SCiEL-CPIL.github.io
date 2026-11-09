def prime_number_theorem_partial_function():
    K = 200000
    mu = _compute_mobius(K)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    return domain, values
