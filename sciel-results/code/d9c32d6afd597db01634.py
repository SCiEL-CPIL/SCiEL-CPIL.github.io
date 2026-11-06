import math

def construct():
    # For n=255=3*5*17, the optimal pairwise non-coprime set containing 255
    # is all multiples of the smallest prime factor (3), giving 85 elements.
    # This is provably optimal: any such set must have all elements sharing
    # a common factor with 255, and multiples of 3 maximize cardinality.
    result = [i for i in range(1, 256) if i % 3 == 0]
    # 255 = 3*85, so 255 is included
    return result
