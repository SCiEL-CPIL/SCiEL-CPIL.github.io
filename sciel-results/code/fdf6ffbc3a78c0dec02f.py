import itertools
import random

def is_sidon(subset):
    """
    Check if a subset is a Sidon set.
    A Sidon set requires all unordered pair sums (including diagonals a+a) to be unique.
    """
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """
    Find the size of the largest Sidon subset of S.
    S is a list of distinct integers.
    """
    n = len(S)
    # Check from largest possible size down to 1
    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    # The densest set of 12 distinct integers is range(12) i.e., 0 to 11.
    # Dense sets tend to have smaller maximum Sidon subsets due to more sum collisions.
    # Random searches in larger ranges (like [0, 30]) are less dense and thus 
    # likely to support larger Sidon subsets.
    # Therefore, the arithmetic progression 0..11 is a strong candidate for minimizing
    # the largest Sidon subset size.
    
    return list(range(12))
