import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a set of 31 distinct integers in [1, 256] such that
    subset sums of different cardinalities are disjoint.
    
    The maximum possible size is 31.
    We use an arithmetic progression starting at 226.
    x_i = 226 + (i-1) for i=1..31.
    x_31 = 256.
    
    Condition for AP: x_1 > c * (k - 1 - c) for all c in 1..k-1.
    Max of c * (30 - c) is 225 (at c=15).
    So x_1 >= 226 is sufficient.
    """
    k = 31
    x_1 = 226
    return list(range(x_1, x_1 + k))
# EVOLVE-BLOCK-END
