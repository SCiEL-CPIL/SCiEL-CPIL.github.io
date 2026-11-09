def construct():
    """
    Returns the optimal set of 12 distinct positive integers with unique
    subset sums, minimizing the largest integer.
    
    The optimal set is the binary sequence [1, 2, 4, ..., 2048], where the
    largest element is 2^11 = 2048.
    
    Proof of optimality:
    Let the sorted sequence be a_1 < a_2 < ... < a_n.
    For all subset sums to be distinct, a_k must be greater than the sum of
    all previous elements: a_k > sum_{i=1}^{k-1} a_i.
    This implies a_k >= 2^{k-1}.
    For n=12, a_12 >= 2^11 = 2048.
    The binary sequence achieves this bound.
    """
    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
