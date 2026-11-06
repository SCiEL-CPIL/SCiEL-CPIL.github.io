import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    MIN_N = 1000
    MAX_N = 1000000
    W = 24
    # primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Precompute smooth part (factor composed of primes < 24) for 1..MAX_N+W
    Nmax = MAX_N + W
    smooth = [1] * (Nmax + 1)
    
    # For each number from 2 to Nmax, compute its smooth part
    # by dividing out all small prime factors
    for i in range(2, Nmax + 1):
        n = i
        s = 1
        for p in primes:
            while n % p == 0:
                s *= p
                n //= p
            if n == 1:
                break
        smooth[i] = s

    # Now slide window of size W over n in [MIN_N, MAX_N]
    # For window starting at n, consider smooth[n+1] .. smooth[n+W]
    # Count distinct values.
    # Use a frequency dict for the sliding window.

    # Initialize window for n = MIN_N: indices MIN_N+1 .. MIN_N+W
    freq = defaultdict(int)
    for i in range(MIN_N + 1, MIN_N + W + 1):
        freq[smooth[i]] += 1

    best_count = len(freq)
    best_n = MIN_N

    # Slide
    for n in range(MIN_N + 1, MAX_N + 1):
        # remove smooth[n] (which was smooth[(n-1)+1])
        out_val = smooth[n]
        freq[out_val] -= 1
        if freq[out_val] == 0:
            del freq[out_val]
        # add smooth[n+W]
        in_val = smooth[n + W]
        freq[in_val] += 1

        cur_count = len(freq)
        if cur_count < best_count:
            best_count = cur_count
            best_n = n
            if best_count == 1:
                # Can't do better than 1
                break

        if time.time() - start_time > DEADLINE:
            break

    return best_n
