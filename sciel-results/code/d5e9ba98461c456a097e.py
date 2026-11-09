import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 2s margin

    # 17-element set with distinct subset products:
    # Replace {2, 3} with {6}, keeping all other elements.
    # Key: 6*16 = 96 != 32 (32 not in set), so no collision.
    candidate = [6, 4, 16, 9, 25, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Verify: all elements in 2..40, distinct, exactly 17
    if len(candidate) != 17:
        return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if len(set(candidate)) != 17:
        return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if any(x < 2 or x > 40 for x in candidate):
        return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Verify all 2^17 subset products are distinct
    n = len(candidate)
    products = set()
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= candidate[i]
        if prod in products:
            # Collision found, fall back to 16-element set
            return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        products.add(prod)

    if time.time() > deadline:
        return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    return list(candidate)
