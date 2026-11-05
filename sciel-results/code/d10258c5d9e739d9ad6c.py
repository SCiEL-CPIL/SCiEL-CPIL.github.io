def construct():
    import time
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    from math import isqrt
    import numpy as np

    # Generate all primes <= 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [p for p in range(n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)

    if n <= 1:
        return primes

    # DP: For each prime index i, states[i] is a list of (last_gap, length, prev_index)
    # Pareto-optimal: sorted by last_gap ascending, length strictly increasing
    # (0, 1, -1) represents a sequence of length 1 (just primes[i] itself)
    states = [[] for _ in range(n)]
    states[0].append((0, 1, -1))

    best_length = 1
    best_end = 0

    for i in range(1, n):
        if elapsed() > DEADLINE:
            break

        pi = primes[i]
        
        # Vectorized gap computation for all j < i
        gaps = pi - primes_arr[:i]
        
        # For each j < i, find the best extension
        # We need the rightmost entry in states[j] with last_gap <= gaps[j]
        # that gives the maximum length
        
        best_candidate = (0, 1, -1)  # just primes[i] alone
        best_len = 1
        
        for j in range(i):
            if elapsed() > DEADLINE:
                break

            gap = int(gaps[j])
            sj = states[j]
            if not sj:
                continue

            # Binary search: rightmost entry in sj with last_gap <= gap
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            # lo-1 is the last valid index (last_gap <= gap)
            if lo > 0:
                last_gap_j, len_j, prev_j = sj[lo - 1]
                new_len = len_j + 1
                if new_len > best_len:
                    best_len = new_len
                    best_candidate = (gap, new_len, j)

        # Build Pareto-optimal states for i
        # We need all (gap, length) pairs where length is strictly increasing with gap
        # Collect all valid extensions
        candidates = [(0, 1, -1)]  # standalone
        
        for j in range(i):
            if elapsed() > DEADLINE:
                break

            gap = int(gaps[j])
            sj = states[j]
            if not sj:
                continue

            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 0:
                last_gap_j, len_j, prev_j = sj[lo - 1]
                candidates.append((gap, len_j + 1, j))

        # Filter to Pareto-optimal: sort by gap, keep strictly increasing length
        if candidates:
            candidates.sort(key=lambda x: (x[0], -x[1]))
            filtered = []
            max_len = 0
            for gap, length, prev in candidates:
                if length > max_len:
                    filtered.append((gap, length, prev))
                    max_len = length
            states[i] = filtered

            if filtered:
                last_len = filtered[-1][1]
                if last_len > best_length:
                    best_length = last_len
                    best_end = i

    # Reconstruct the sequence ending at best_end
    if best_length <= 1:
        return [primes[best_end]]

    seq = []
    idx = best_end
    while idx != -1:
        seq.append(primes[idx])
        sj = states[idx]
        if not sj:
            break
        # Find the state entry that gives the best length for this index
        target_len = max(e[1] for e in sj)
        # Find the entry with this length (use the one with smallest gap for that length)
        prev = -1
        for gap_val, length_val, prev_val in sj:
            if length_val == target_len:
                prev = prev_val
                break
        idx = prev

    seq.reverse()

    # Multi-start local search: try greedy extension from multiple candidate endpoints
    if elapsed() < DEADLINE - 5:
        def greedy_extend(start_seq):
            """Greedy extension: always pick the smallest valid next prime."""
            extended = list(start_seq)
            while elapsed() < DEADLINE - 5:
                if len(extended) >= 2:
                    last_gap = extended[-1] - extended[-2]
                else:
                    last_gap = 0
                last_prime = extended[-1]
                next_prime = None
                for p in primes:
                    if p > last_prime and (p - last_prime) >= last_gap:
                        next_prime = p
                        break
                if next_prime is None:
                    break
                extended.append(next_prime)
            return extended

        def greedy_extend_largest(start_seq):
            """Greedy extension: pick the largest valid next prime (maximizes future gap room)."""
            extended = list(start_seq)
            while elapsed() < DEADLINE - 5:
                if len(extended) >= 2:
                    last_gap = extended[-1] - extended[-2]
                else:
                    last_gap = 0
                last_prime = extended[-1]
                next_prime = None
                for p in reversed(primes):
                    if p > last_prime and (p - last_prime) >= last_gap:
                        next_prime = p
                        break
                if next_prime is None:
                    break
                extended.append(next_prime)
            return extended

        best_seq = list(seq)
        best_len = len(seq)

        # Try extension from the current best
        for ext_fn in [greedy_extend, greedy_extend_largest]:
            ext = ext_fn(seq)
            if len(ext) > best_len:
                best_len = len(ext)
                best_seq = ext

        # Try other candidate endpoints: for each index i with good states,
        # reconstruct a sequence and try extending
        if elapsed() < DEADLINE - 15:
            # Collect top candidates by length
            candidates_to_try = []
            for i in range(n):
                sj = states[i]
                if sj:
                    max_l = sj[-1][1]
                    if max_l >= best_len - 2:  # within 2 of best
                        candidates_to_try.append((max_l, i))
            # Sort by length descending, try a limited number
            candidates_to_try.sort(key=lambda x: -x[0])
            tried = 0
            for max_l, i in candidates_to_try:
                if elapsed() > DEADLINE - 10:
                    break
                if tried >= 20:
                    break
                # Reconstruct sequence ending at i
                seq_i = []
                idx = i
                while idx != -1:
                    seq_i.append(primes[idx])
                    sj = states[idx]
                    if not sj:
                        break
                    target_len = max(e[1] for e in sj)
                    prev = -1
                    for gap_val, length_val, prev_val in sj:
                        if length_val == target_len:
                            prev = prev_val
                            break
                    idx = prev
                seq_i.reverse()
                tried += 1
                for ext_fn in [greedy_extend, greedy_extend_largest]:
                    ext = ext_fn(seq_i)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best_seq = ext

        if len(best_seq) > len(seq):
            seq = best_seq

    return seq
