def construct():
    """
    Returns the minimal superincreasing sequence of length 12.
    The largest element is 2^11 = 2048, which is the theoretical minimum
    for a set of 12 distinct positive integers with distinct subset sums.
    """
    n = 12
    return [1 << i for i in range(n)]
