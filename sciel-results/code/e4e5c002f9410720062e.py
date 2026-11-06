import random
import time

def construct():
    n = 4096
    # Algebraic construction: Powers of 4
    # Sums of 3 elements have coefficients 0..3 in base 4.
    # Since 3 < 4, representations are unique.
    # 4^0, 4^1, ..., 4^m
    # Max element <= 4096
    # 4^6 = 4096
    # So we can use 4^0 to 4^6
    base = 4
    max_val = n
    elements = []
    val = 1
    while val <= max_val:
        elements.append(val)
        val *= base
    
    return elements
