import time

_SMALL_PRIMES = [2,3,5,7,11,13,17,19,23]

def _small_factor(x):
    f = 1
    for p in _SMALL_PRIMES:
        while x % p == 0:
            f *= p
            x //= p
    return f

def construct():
    deadline = time.time() + 80.0
    MAXN = 1000000
    # spf for indices 1..MAXN+24
    N = MAXN + 24
    spf = [1]*(N+1)
    for x in range(1, N+1):
        t = x
        f = 1
        for p in _SMALL_PRIMES:
            while t % p == 0:
                f *= p
                t //= p
        spf[x] = f
        if time.time() > deadline - 20:
            break
    # scan windows
    best_n = 1000
    best_count = None
    for s in range(1000, MAXN+1):
        if time.time() > deadline:
            break
        window = spf[s+1:s+25]
        c = len(set(window))
        if best_count is None or c < best_count:
            best_count = c
            best_n = s
        if c <= 2:  # can't do better than 1; 2 is very good, keep scanning a bit
            if c == 1:
                break
    return best_n
