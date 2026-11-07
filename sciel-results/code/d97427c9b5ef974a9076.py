def construct():
    # The maximum number of multiplicatively independent integers
    # from the set {2, ..., 40} is equal to the number of primes
    # in that range, because the exponent vectors of the numbers
    # must be linearly independent in the vector space over Q
    # spanned by the prime exponents.
    # The primes in [2, 40] are:
    # 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # There are 12 such primes.
    # Returning these primes guarantees distinct subset products
    # by the Fundamental Theorem of Arithmetic.
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
