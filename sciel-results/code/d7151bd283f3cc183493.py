import time
from math import gcd

def construct():
    """
    Returns a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    
    The set consists of powers of 2 (excluding 1) and powers of 3 (excluding 1).
    S = {2, 4, 8, 16, 32, 64} U {3, 9, 27, 81}
    Size = 10.
    """
    # Powers of 2 up to 128, excluding 1
    p2 = [2**i for i in range(1, 7)]  # 2, 4, 8, 16, 32, 64
    
    # Powers of 3 up to 128, excluding 1
    p3 = [3**i for i in range(1, 5)]  # 3, 9, 27, 81
    
    S = sorted(p2 + p3)
    
    return S
