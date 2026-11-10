"""Program finding long increasing prime sequences with nondecreasing gaps <= 10000."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import isqrt
    import numpy as np

    deadline = time.time() + 85  # 5s safety margin under 90s wall-clock
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

    # P[i] = prefix max of dp[0..i-1][i], with argmax
    # P_val[i][k] = max(dp[x][i] for x in [0, k]) for k < i
    # P_arg[i][k] = argmax x in [0, k] achieving that max
    P_val = [None] * m
    P_arg = [None] * m

    best_len = 0
    best_i = -1
    best_j = -1

    for j in range(m):
        if time.time() > deadline:
            break
        b = primes[j]
        
        if j > 0:
            t_vals = 2 * primes[:j] - b
            lo_indices = np.searchsorted(primes, t_vals, side='left')
            
            for i in range(j):
                lo = int(lo_indices[i])
                # We need k < i AND primes[k] <= 2*primes[i]-primes[j]
                # lo = first index where primes[lo] >= 2*primes[i]-primes[j]
                # So valid k are in [0, min(i-1, lo-1)]
                upper = min(i - 1, lo - 1)
                if upper >= 0:
                    max_prev = int(P_val[i][upper])
                    dp[i, j] = 2 + max_prev
                    amax[i, j] = P_arg[i][upper]
                else:
                    dp[i, j] = 2
                    amax[i, j] = -1
                
                if dp[i, j] > best_len:
                    best_len = int(dp[i, j])
                    best_i = i
                    best_j = j
        
        # Compute P[j] = prefix max of dp[0..j-1][j]
        if j > 0:
            pv = np.zeros(j, dtype=np.int16)
            pa = np.zeros(j, dtype=np.int16)
            pv[0] = dp[0, j]
            pa[0] = 0
            for k in range(1, j):
                if dp[k, j] >= pv[k-1]:
                    pv[k] = dp[k, j]
                    pa[k] = k
                else:
                    pv[k] = pv[k-1]
                    pa[k] = pa[k-1]
            P_val[j] = pv
            P_arg[j] = pa

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
