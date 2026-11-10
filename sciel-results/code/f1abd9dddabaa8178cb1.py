def construct():
    # Known optimal set of 16 distinct integers in [2, 40]
    # with distinct subset products.
    # Based on prime powers with distinct exponent subset sums.
    return [
        2, 4, 16,   # Powers of 2 (exponents 1, 2, 4)
        3, 9,       # Powers of 3 (exponents 1, 2)
        5, 25,      # Powers of 5 (exponents 1, 2)
        7, 11, 13, 17, 19, 23, 29, 31, 37  # Other primes (exponent 1)
    ]
