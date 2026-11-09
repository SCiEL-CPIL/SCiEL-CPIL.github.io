import time
from collections import Counter

def _smooth_part(m, primes):
    """Extract the part of m composed only of the given primes."""
    result = 1
    for p in primes:
        while m % p == 0:
            result *= p
            m //= p
    return result

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_num = 1000024

    # Precompute smooth parts for all relevant numbers
    smooth = [0] * (max_num + 1)
    for i in range(1001, max_num + 1):
        smooth[i] = _smooth_part(i, primes)

    # Sliding window with frequency counter
    best_n = 1000
    best_count = float('inf')

    # Initialize window for n=1000: indices 1001 to 1024
    freq = Counter()
    for i in range(1001, 1025):
        freq[smooth[i]] += 1

    distinct_count = len(freq)
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = 1000

    for n in range(1001, 1000001):
        if time.time() > deadline:
            break
        # Remove old left element, add new right element
        old_val = smooth[n]       # (n-1)+1
        new_val = smooth[n + 24]  # (n-1)+25

        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct_count -= 1

        if new_val in freq:
            freq[new_val] += 1
        else:
            freq[new_val] = 1
            distinct_count += 1

        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n

    return best_n
