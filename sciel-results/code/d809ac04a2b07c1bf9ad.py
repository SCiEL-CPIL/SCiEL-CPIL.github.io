def construct():
    """
    Constructs a set of n=12 distinct positive integers such that all 
    subset sums are distinct, minimizing the largest integer.
    
    The optimal strategy is to use a superincreasing sequence where each 
    element is strictly greater than the sum of all previous elements. 
    To minimize the largest element, we choose the smallest possible 
    integer at each step that satisfies this condition.
    
    This generates the sequence 1, 2, 4, 8, ..., which are powers of 2.
    The largest element will be 2^(n-1).
    """
    n = 12
    # The minimal largest element for n distinct positive integers with 
    # distinct subset sums is 2^(n-1). The sequence is powers of 2.
    return [2**i for i in range(n)]
