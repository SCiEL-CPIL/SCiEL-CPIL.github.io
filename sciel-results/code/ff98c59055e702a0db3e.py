def construct():
    """
    Constructs a set of n=12 distinct positive integers such that all subset sums are distinct.
    This requires the set to be superincreasing: a_k > sum(a_1...a_{k-1}).
    To minimize the largest element a_n, we choose the smallest possible values for each a_k.
    a_1 = 1
    a_k = 1 + sum(a_1...a_{k-1}) = 2 * sum(a_1...a_{k-1}) = 2^k (if 0-indexed, a_i = 2^i)
    The optimal set is {1, 2, 4, ..., 2^(n-1)}.
    The largest element is 2^(n-1).
    """
    n = 12
    # Generate powers of 2: 2^0, 2^1, ..., 2^(n-1)
    return [1 << i for i in range(n)]
