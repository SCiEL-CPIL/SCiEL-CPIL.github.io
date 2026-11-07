import time

P = {'n': 128}

def construct():
    n = P['n']
    
    # Construction: L = {16, ..., 32}, U = {65, ..., 128}
    # L is weakly sum-free (min sum 33 > max L 32)
    # U is sum-free (min sum 130 > n)
    # L+L is in [33, 64], disjoint from L and U
    # L+U is in [81, 160]. Intersection with [1, 128] is [81, 128].
    #   For x in L, y in U: x+y in U (since min 16+65=81 > 65).
    #   Wait, if x+y in U, does it cause a problem?
    #   Triple x, y, z with x in L, y, z in U.
    #   Sums: x+y, x+z, y+z.
    #   y+z > 128, so y+z not in S.
    #   So triple is safe.
    #   Triple x, y, z with x, y in L, z in U.
    #   Sums: x+y, x+z, y+z.
    #   x+y in [33, 64]. Not in L (max 32) and not in U (min 65).
    #   So x+y not in S.
    #   So triple is safe.
    
    result = []
    # Add L
    for i in range(16, 33):
        result.append(i)
    # Add U
    for i in range(65, 129):
        result.append(i)
        
    return result
