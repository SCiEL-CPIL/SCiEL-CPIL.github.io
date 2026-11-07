import time

# EVOLVE-BLOCK-START
def construct():
    # The optimal set size is 31, using arithmetic progression [226, 227, ..., 256]
    # This set satisfies the condition that for any two subsets of different cardinalities,
    # their sums differ.
    # Specifically, max sum of k-1 elements < min sum of k elements for all k.
    k = 31
    x_1 = 226
    result = list(range(x_1, x_1 + k))
    
    # Basic validation
    assert len(set(result)) == len(result)
    assert all(1 <= x <= 256 for x in result)
    
    return result
# EVOLVE-BLOCK-END
