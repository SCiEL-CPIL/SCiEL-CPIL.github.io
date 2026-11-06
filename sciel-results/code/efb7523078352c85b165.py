import time
from math import gcd

def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255,
    such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    
    The maximum cardinality is achieved by taking all multiples of 3 in [1, 255].
    Size = 255 // 3 = 85.
    255 is a multiple of 3.
    Any two multiples of 3 share a factor of 3, so gcd > 1.
    """
    # The set of all multiples of 3 in [1, 255]
    result = list(range(3, 256, 3))
    
    # Ensure 255 is included (it is, 255 % 3 == 0)
    # Ensure non-empty (it is)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Verify
    valid = True
    if 255 not in res:
        valid = False
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if gcd(res[i], res[j]) == 1:
                valid = False
                break
    print("Valid:", valid)
