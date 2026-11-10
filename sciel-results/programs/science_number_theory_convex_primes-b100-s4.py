"""Program finding long increasing prime sequences with nondecreasing gaps <= 10000."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import isqrt
    import numpy as np

    deadline = time.time() + 80  # 10s safety margin under 90s wall-clock
    t0 = time.time()

    # --- Sieve primes up to 10000 ---
    n_max = 10000
    is_prime = bytearray(b'\x01') * (n_max + 1)
    is_prime[0] = 0
    if n_max >= 1:
        is_prime[1] = 0
    for i in range(2, isqrt(n_max) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(range(i*i, n_max+1, i)))
    primes = np.array([i for i in range(2, n_max + 1) if is_prime[i]], dtype=np.int32)
    m = len(primes)

    # dp[i][j] = max length of valid chain ending at primes[i], primes[j] (i < j)
    # amax[i][j] = the k that achieved the max (for reconstruction)
    dp = np.zeros((m, m), dtype=np.int16)
    amax = np.full((m, m), -1, dtype=np.int16)

    # S[i] = suffix max of dp[0..i-1][i], with argmax
    # S_val[i][k] = max(dp[x][i] for x in [k, i-1])
    # S_arg[i][k] = argmax x in [k, i-1] achieving that max
    S_val = [None] * m
    S_arg = [None] * m

    best_len = 0
    best_i = -1
    best_j = -1

    for j in range(m):
        if time.time() > deadline:
            break
        b = primes[j]
        for i in range(j):
            a = primes[i]
            t = 2 * a - b  # need primes[k] >= t for k < i
            # Binary search for smallest k in [0, i-1] with primes[k] >= t
            if t <= primes[0]:
                lo = 0
            elif t > primes[i-1]:
                lo = i  # no valid k
            else:
                lo = 0
                hi = i
                while lo < hi:
                    mid = (lo + hi) // 2
                    if primes[mid] >= t:
                        hi = mid
                    else:
                        lo = mid + 1
            if lo < i:
                max_prev = S_val[i][lo]
                dp[i, j] = 2 + max_prev
                amax[i, j] = S_arg[i][lo]
            else:
                dp[i, j] = 2
                amax[i, j] = -1
            if dp[i, j] > best_len:
                best_len = dp[i, j]
                best_i = i
                best_j = j
        # Compute S[j] = suffix max of dp[0..j-1][j]
        if j > 0:
            sv = np.zeros(j, dtype=np.int16)
            sa = np.zeros(j, dtype=np.int16)
            sv[j-1] = dp[j-1, j]
            sa[j-1] = j-1
            for k in range(j-2, -1, -1):
                if dp[k, j] >= sv[k+1]:
                    sv[k] = dp[k, j]
                    sa[k] = k
                else:
                    sv[k] = sv[k+1]
                    sa[k] = sa[k+1]
            S_val[j] = sv
            S_arg[j] = sa

    if best_len <= 2:
        return [int(primes[0]), int(primes[1])]

    # Reconstruct chain
    chain = [int(primes[best_j]), int(primes[best_i])]
    ci = best_i
    cj = best_j
    clen = best_len

    while clen > 2:
        k = int(amax[ci, cj])
        if k < 0:
            break
        chain.append(int(primes[k]))
        cj = ci
        ci = k
        clen -= 1

    chain.reverse()

    # Verify
    prime_set = set(primes.tolist())
    valid = (len(chain) >= 2 and
             all(chain[i+1] > chain[i] for i in range(len(chain)-1)) and
             all(p in prime_set for p in chain) and
             all(chain[i+1] - chain[i] >= chain[i] - chain[i-1] for i in range(1, len(chain)-1)) and
             chain[-1] <= 10000)

    if valid:
        return chain
    else:
        # Fallback: greedy
        values = []
        for p in range(2, 10001):
            if all(p % d for d in range(2, isqrt(p) + 1)):
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
        return values

# EVOLVE-BLOCK-END
