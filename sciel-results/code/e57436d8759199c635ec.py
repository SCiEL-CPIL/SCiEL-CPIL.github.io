def construct():
    # Polynomial P(x) = 1 + x + x^2 + ... + x^11
    # This has exactly 12 non-zero terms, all with coefficient 1.
    # P(x)^2 will have coefficients 1, 2, ..., 12, ..., 2, 1.
    # The number of non-zero terms in P(x)^2 is 23.
    return [1] * 12
