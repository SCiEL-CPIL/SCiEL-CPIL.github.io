def construct():
    # The condition for all 2^n subset sums to be distinct is that the set,
    # when sorted as a_0 < a_1 < ... < a_{n-1}, satisfies a_k > sum(a_0...a_{k-1}).
    # To minimize the largest element a_{n-1}, we must minimize all preceding elements.
    # a_0 >= 1
    # a_1 > a_0 => a_1 >= 2
    # a_2 > a_0 + a_1 => a_2 >= 4
    # ...
    # a_k >= 2^k
    # Thus the optimal set is {1, 2, 4, ..., 2^(n-1)}.
    # For n=12, the largest element is 2^11 = 2048.
    n = 12
    return [1 << i for i in range(n)]
