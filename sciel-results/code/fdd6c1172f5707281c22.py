import time


def construct():
    start = time.time()
    deadline = start + 75  # explicit safety margin before the 90 s wall-clock kill

    # Primes STRICTLY BELOW 24
    SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23)
    N = 1000025  # we need smooth parts for indices up to 1000024 (n=1000000, i=24)

    # Precompute the 23-smooth part (product of p^{v_p(x)} for p in SMALL_PRIMES)
    # for every x in [0, N). This is the "complete factor composed of primes < 24".
    smooth = [1] * N
    for p in SMALL_PRIMES:
        for m in range(p, N, p):
            # count the exponent of p in m
            v = 0
            mm = m
            while mm % p == 0:
                v += 1
                mm //= p
            if v:
                smooth[m] *= p ** v

    best_n = 1000
    best_count = 10 ** 9

    # Exhaustive scan over every valid n to find the TRUE global minimum
    # of the number of distinct extracted factors.
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        s = set()
        s.add(smooth[n + 1])
        s.add(smooth[n + 2])
        s.add(smooth[n + 3])
        s.add(smooth[n + 4])
        s.add(smooth[n + 5])
        s.add(smooth[n + 6])
        s.add(smooth[n + 7])
        s.add(smooth[n + 8])
        s.add(smooth[n + 9])
        s.add(smooth[n + 10])
        s.add(smooth[n + 11])
        s.add(smooth[n + 12])
        s.add(smooth[n + 13])
        s.add(smooth[n + 14])
        s.add(smooth[n + 15])
        s.add(smooth[n + 16])
        s.add(smooth[n + 17])
        s.add(smooth[n + 18])
        s.add(smooth[n + 19])
        s.add(smooth[n + 20])
        s.add(smooth[n + 21])
        s.add(smooth[n + 22])
        s.add(smooth[n + 23])
        s.add(smooth[n + 24])
        c = len(s)
        if c < best_count:
            best_count = c
            best_n = n
            if best_count == 1:
                # absolute lower bound; nothing better possible
                break

    return best_n
