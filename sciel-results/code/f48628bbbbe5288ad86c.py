"""Optimal DP for longest nondecreasing-gap prime subsequence <=10000."""
import time
import bisect

def _sieve_primes(limit):
    """Return list of all primes <= limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    primes = _sieve_primes(10000)
    n = len(primes)

    if n < 2:
        return primes[:]

    # DP state: dp[j][i] = length of longest valid subsequence ending with primes[j], primes[i]
    # where j < i and gaps are nondecreasing.
    # 
    # For each i, we need sorted_struct[j] for all j < i to answer queries.
    # sorted_struct[j] = (sorted_gaps, prefix_maxes, prefix_argmax)
    #   where for each k < j, we have (primes[j]-primes[k], dp[k][j], k) sorted by gap.
    #
    # Key optimization: We don't need to store all dp_table entries.
    # We only need the current layer's dp values to build sorted_struct[i].
    # For reconstruction, we store argmax pointers compactly.

    sorted_struct = [None] * n  # (sorted_gaps, prefix_maxes, prefix_argmax)
    
    # For reconstruction: store argmax[i][j] = k for the best predecessor
    # Use a list of arrays, one per i
    argmax_layers = [None] * n

    best_length = 2
    best_i = 1
    best_j = 0

    for i in range(1, n):
        # Compute dp[j][i] for all j < i
        cur_dp = [0] * i
        cur_argmax = [-1] * i

        for j in range(i):
            gap = primes[i] - primes[j]

            if sorted_struct[j] is not None:
                sorted_gaps, prefix_maxes, prefix_argmax = sorted_struct[j]
                idx = bisect.bisect_right(sorted_gaps, gap) - 1
                if idx >= 0:
                    val = prefix_maxes[idx] + 1
                    cur_dp[j] = val
                    cur_argmax[j] = prefix_argmax[idx]
                else:
                    cur_dp[j] = 2
                    cur_argmax[j] = -1
            else:
                cur_dp[j] = 2
                cur_argmax[j] = -1

            if cur_dp[j] > best_length:
                best_length = cur_dp[j]
                best_i = i
                best_j = j

        # Store argmax for reconstruction
        argmax_layers[i] = cur_argmax

        # Build structure for i: for all k < i, (primes[i]-primes[k], dp[k][i], k)
        if i > 0:
            # Build list of (gap, dp_val, k) for k in range(i)
            # Use list comprehension for speed
            pi = primes[i]
            pairs = [(pi - primes[k], cur_dp[k], k) for k in range(i)]

            pairs.sort(key=lambda x: x[0])
            m = len(pairs)
            sorted_gaps = [p[0] for p in pairs]
            prefix_maxes = [0] * m
            prefix_argmax = [0] * m
            prefix_maxes[0] = pairs[0][1]
            prefix_argmax[0] = pairs[0][2]
            for idx in range(1, m):
                if pairs[idx][1] > prefix_maxes[idx - 1]:
                    prefix_maxes[idx] = pairs[idx][1]
                    prefix_argmax[idx] = pairs[idx][2]
                else:
                    prefix_maxes[idx] = prefix_maxes[idx - 1]
                    prefix_argmax[idx] = prefix_argmax[idx - 1]

            sorted_struct[i] = (sorted_gaps, prefix_maxes, prefix_argmax)

        # Check time periodically
        if i % 50 == 0 and time.time() > deadline:
            break

    # Reconstruct the best subsequence
    if best_j < 0:
        return primes[:1]

    seq = [primes[best_i], primes[best_j]]
    cur_i = best_i
    cur_j = best_j

    while cur_j >= 0 and argmax_layers[cur_i][cur_j] >= 0:
        k = argmax_layers[cur_i][cur_j]
        seq.append(primes[k])
        cur_i = cur_j
        cur_j = k

    seq.reverse()

    return seq
