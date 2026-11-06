import time
import math

def construct():
    start = time.time()
    deadline = start + 80.0

    N = 1000000
    LO = 10000
    # n must be in [10000, 1000000], so n+1 can be 1000001
    limit = N + 1

    # Build SPF up to limit
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, math.isqrt(limit) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Compute powerful part for each number
    pp = [1] * (limit + 1)
    # pp[0] = 0, pp[1] = 1
    for n in range(2, limit + 1):
        temp = n
        result = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                result *= p ** e
        pp[n] = result

    if time.time() > deadline:
        return LO

    best_n = LO
    best_val = pp[LO] * pp[LO + 1]
    
    # Scan for max
    for n in range(LO, N):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
        if time.time() > deadline:
            break

    return best_n
