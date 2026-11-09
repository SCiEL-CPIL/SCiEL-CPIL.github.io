import time

def construct():
    """
    Returns a maximal subset of [1, 256] such that for all a < b <= c in the set,
    a does not divide b + c.
    
    The set [171, 256] is valid because for any a in this range, 3*a > 2*256 = 512.
    For any b, c > a in the set, b + c > 2*a.
    The next multiple of a is 3*a, which is > 512.
    Since b + c <= 2*256 = 512, b + c cannot be a multiple of a.
    Thus, the property holds.
    Size is 256 - 171 + 1 = 86.
    """
    n = 256
    # k is the smallest integer such that 3*k > 2*n
    # 3*k > 512 => k > 170.66 => k = 171
    k = 171
    return list(range(k, n + 1))
