import math
import time

def construct():
    # The maximum size of a pairwise non-coprime subset of [1, 255] containing 255
    # is determined by the prime factor of 255 with the most multiples.
    # 255 = 3 * 5 * 17.
    # Multiples of 3: 85
    # Multiples of 5: 51
    # Multiples of 17: 15
    # The set of all multiples of 3 is a valid clique of size 85.
    # It is impossible to form a larger clique containing 255.
    return list(range(3, 256, 3))
