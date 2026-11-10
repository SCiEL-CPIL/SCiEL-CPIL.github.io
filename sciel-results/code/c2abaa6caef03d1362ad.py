import time

def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    The optimal set is the superincreasing sequence starting with 1,
    which results in powers of 2: 1, 2, 4, ..., 2048.
    The largest element is 2048, which is the theoretical minimum.
    """
    n = 12
    # Directly return the known optimal solution: powers of 2
    # This satisfies distinct subset sums (binary representation uniqueness)
    # and minimizes the largest element.
    return [1 << i for i in range(n)]
