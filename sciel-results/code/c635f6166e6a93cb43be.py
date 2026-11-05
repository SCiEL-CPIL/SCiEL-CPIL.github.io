def construct():
    """
    Returns the set of all multiples of 3 in [1, 255].
    This is the maximum cardinality subset containing 255
    where all pairs have gcd > 1.
    Size: 85.
    """
    return list(range(3, 256, 3))
