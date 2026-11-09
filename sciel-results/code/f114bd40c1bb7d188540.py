import time

def construct():
    """
    Find a set of n=12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    The theoretical minimum for the largest element in a set of n distinct
    positive integers with distinct subset sums is 2^(n-1).
    For n=12, this is 2^11 = 2048.
    The set of powers of 2 {1, 2, 4, ..., 2048} achieves this bound.
    """
    n = 12
    # The optimal solution is the powers of 2
    return [1 << i for i in range(n)]
