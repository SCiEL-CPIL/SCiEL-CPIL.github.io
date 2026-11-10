"""Program finding long increasing prime sequences with nondecreasing gaps <= 10000."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import bisect
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
    primes_list = primes.tolist()
    prime_set = set(primes_list)

    # dp[i] is a dict: gap -> max length of valid chain ending at primes[i] with last gap = gap
    # For each i, we also maintain sorted_gaps and prefix_max for efficient range-max queries
    # prefix_max[k] = max of dp[i][g] for all gaps g in sorted_gaps[0..k]
    # prefix_arg[k] = the index in sorted_gaps that achieves the prefix max (for reconstruction)

    # Actually, for reconstruction we need to know:
    #   dp[i][gap] = best length ending at i with last gap = gap
    #   parent[i][gap] = (j, prev_gap) such that dp[j][prev_gap] + 1 = dp[i][gap]
    #
    # But we also need to handle the case where the best chain ending at i with last gap = gap
    # comes from a specific j and prev_gap.
    #
    # Let's use:
    #   dp[i] = dict: gap -> (length, j, prev_gap)
    #   For prefix max queries on j, we need sorted gaps and prefix max of lengths.

    dp = [None] * m  # dp[i] = dict: gap -> (length, j, prev_gap)
    # prefix_data[i] = (sorted_gaps, prefix_max_vals, prefix_max_arg_idx)
    #   where prefix_max_arg_idx[k] = index in sorted_gaps achieving prefix max up to k
    prefix_data = [None] * m

    best_len = 1
    best_i = 0
    best_gap = 0

    for i in range(m):
        if time.time() > deadline:
            break

        b = primes_list[i]
        dp[i] = {}

        # Base case: chain of length 1 (no previous gap)
        # We'll handle this implicitly: any chain of length >= 2 has a last gap.
        # For the first element, we can think of it as having "gap" = 0 with length 1.

        if i == 0:
            dp[i][0] = (1, -1, -1)
            best_len = 1
            best_i = 0
            best_gap = 0
        else:
            # Try each j < i as the previous element
            for j in range(i):
                gap = b - primes_list[j]
                if gap <= 0:
                    continue

                # Query prefix_data[j] for max length with gap' <= gap
                if prefix_data[j] is not None:
                    sorted_gaps, prefix_max_vals, prefix_max_arg_idx = prefix_data[j]
                    # Find rightmost index where sorted_gaps[idx] <= gap
                    idx = bisect.bisect_right(sorted_gaps, gap)
                    if idx > 0:
                        max_prev_len = prefix_max_vals[idx - 1]
                        prev_gap_idx = prefix_max_arg_idx[idx - 1]
                        prev_gap = sorted_gaps[prev_gap_idx]
                        new_len = max_prev_len + 1
                        if new_len > dp[i].get(gap, (0, -1, -1))[0]:
                            dp[i][gap] = (new_len, j, prev_gap)
                            if new_len > best_len:
                                best_len = new_len
                                best_i = i
                                best_gap = gap
                else:
                    # j has no prefix data (shouldn't happen for j >= 1, but handle j=0)
                    # For j=0, dp[0] has gap=0 with length 1
                    if j == 0:
                        new_len = 2
                        if new_len > dp[i].get(gap, (0, -1, -1))[0]:
                            dp[i][gap] = (new_len, 0, 0)
                            if new_len > best_len:
                                best_len = new_len
                                best_i = i
                                best_gap = gap

            # Also, we need to consider starting a new chain of length 1 at i
            # This is implicitly handled if no j gives a valid transition
            if not dp[i]:
                dp[i][0] = (1, -1, -1)

        # Build prefix_data for i
        if dp[i]:
            items = sorted(dp[i].items())  # list of (gap, (length, j, prev_gap))
            sorted_gaps = [g for g, v in items]
            lengths = [v[0] for g, v in items]
            prefix_max_vals = [0] * len(lengths)
            prefix_max_arg_idx = [0] * len(lengths)
            prefix_max_vals[0] = lengths[0]
            prefix_max_arg_idx[0] = 0
            for k in range(1, len(lengths)):
                if lengths[k] >= prefix_max_vals[k-1]:
                    prefix_max_vals[k] = lengths[k]
                    prefix_max_arg_idx[k] = k
                else:
                    prefix_max_vals[k] = prefix_max_vals[k-1]
                    prefix_max_arg_idx[k] = prefix_max_arg_idx[k-1]
            prefix_data[i] = (sorted_gaps, prefix_max_vals, prefix_max_arg_idx)

    # Reconstruct chain
    if best_len <= 1:
        return [primes_list[0]]

    chain = []
    ci = best_i
    cgap = best_gap

    while True:
        length, j, prev_gap = dp[ci][cgap]
        chain.append(primes_list[ci])
        if j == -1:
            break
        ci = j
        cgap = prev_gap

    chain.reverse()

    # Verify
    valid = (len(chain) >= 1 and
             all(chain[i+1] > chain[i] for i in range(len(chain)-1)) and
             all(p in prime_set for p in chain) and
             all(chain[i+1] - chain[i] >= chain[i] - chain[i-1] for i in range(1, len(chain)-1)) and
             chain[-1] <= 10000)

    if valid:
        return chain
    else:
        # Fallback: simple greedy
        values = [primes_list[0]]
        for p in primes_list[1:]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
        return values

# EVOLVE-BLOCK-END
