import time
import math
import bisect
from bisect import bisect_left

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin for 90s limit

    N_LIMIT = 10000

    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (N_LIMIT + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for p in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = bytearray(len(is_prime[p*p::p]))

    primes = [i for i, val in enumerate(is_prime) if val]
    n = len(primes)

    if n < 2:
        return primes

    # dp[i][j] = length of longest valid subsequence ending with primes[j], primes[i]
    # where j < i (i.e., the last two elements of the subsequence are primes[j] then primes[i])
    # prev[i][j] = the index k such that the subsequence ended with primes[k], primes[j]
    # before adding primes[i]

    dp = [[0] * n for _ in range(n)]
    prev = [[-1] * n for _ in range(n)]

    # suf_val[j][k] = max(dp[k][j], dp[k+1][j], ..., dp[j-1][j])
    # suf_idx[j][k] = index in [k, j-1] that achieves suf_val[j][k]
    suf_val = [None] * n
    suf_idx = [None] * n

    max_len = 0
    best_i = -1
    best_j = -1

    # Iterate over the second-to-last prime index j
    for j in range(1, n):
        if time.time() - t0 > DEADLINE:
            break

        # Pre-compute suffix max for dp[j][0..j-1]
        # suf_val[j][k] = max(dp[j][k], dp[j][k+1], ..., dp[j][j-1])
        # suf_idx[j][k] = index that achieves suf_val[j][k]
        if j > 0:
            sv_arr = [0] * j
            si_arr = [-1] * j
            last_val = dp[j][j-1]
            sv_arr[j-1] = last_val
            si_arr[j-1] = (j-1) if last_val > 0 else -1
            for k in range(j-2, -1, -1):
                v = dp[j][k]
                if v > sv_arr[k+1]:
                    sv_arr[k] = v
                    si_arr[k] = k if v > 0 else -1
                else:
                    sv_arr[k] = sv_arr[k+1]
                    si_arr[k] = si_arr[k+1]
            suf_val[j] = sv_arr
            suf_idx[j] = si_arr

        for i in range(j + 1, n):
            limit = 2 * primes[j] - primes[i]
            k_min = bisect.bisect_left(primes, limit, 0, j)

            if k_min < j:
                sv = suf_val[j][k_min]
                si = suf_idx[j][k_min]
                if sv > 0:
                    length = sv + 1
                    dp[i][j] = length
                    prev[i][j] = si
                else:
                    length = 2
                    dp[i][j] = 2
                    prev[i][j] = -1
            else:
                length = 2
                dp[i][j] = 2
                prev[i][j] = -1

            if length > max_len:
                max_len = length
                best_i = i
                best_j = j

    # Reconstruct the sequence
    if max_len == 0:
        return primes[:1] if n > 0 else []

    seq = [primes[best_i], primes[best_j]]
    cur_i, cur_j = best_i, best_j

    while prev[cur_i][cur_j] != -1:
        k = prev[cur_i][cur_j]
        seq.append(primes[k])
        cur_i, cur_j = cur_j, k

    seq.reverse()

    return seq
