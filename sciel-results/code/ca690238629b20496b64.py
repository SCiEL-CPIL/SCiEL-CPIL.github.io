import time

def construct():
    """
    Returns a subset of [1, 256] such that for all a < b <= c in the set,
    a does not divide b + c.
    Maximizes cardinality.
    
    Strategy: Use the upper tail [k, 256].
    For a in [k, 256], b, c in [k+1, 256].
    Min sum b+c = 2k+2.
    Max sum b+c = 512.
    Multiples of a in this range: 2a, 3a, ...
    Since b > a, b+c > 2a. So 2a is not a sum.
    We need 3a > 512 to ensure no higher multiple is possible.
    3a > 512 => a > 170.66 => a >= 171.
    So [171, 256] is valid.
    Cardinality: 256 - 171 + 1 = 86.
    """
    n = 256
    k = 171
    return list(range(k, n + 1))
