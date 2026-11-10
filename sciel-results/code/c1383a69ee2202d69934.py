import math


def construct():
    # Verified solution: A=[1,7,23], B=[6,18,30]
    # Sums: 7,19,31, 13,25,37, 29,41,53
    # All 9 distinct, all pairwise coprime (25=5^2 shares no factor with any prime in set)
    # This achieves the theoretical maximum of |A+B|=9 for |A|=|B|=3
    return [[1, 7, 23], [6, 18, 30]]
