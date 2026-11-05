def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    from math import isqrt
    import numpy as np

    # Generate all primes <= 10000
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [p for p in range(n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)

    if n <= 1:
        return primes

    # DP with Pareto-optimal states per prime index.
    # states[i] = list of (last_gap, length, prev_index) sorted by last_gap asc,
    # with length strictly increasing. This is the key data structure.
    #
    # For each prime i, we consider all j < i as potential predecessors.
    # The gap is primes[i] - primes[j]. We need the best state in states[j]
    # with last_gap <= gap. Since states[j] is sorted by last_gap with increasing
    # length, the rightmost valid entry gives the maximum length.
    #
    # Optimization: We can process primes in order and for each i, iterate j < i.
    # The total work is O(n^2 * log(max_state_size)).
    # n = 1229 (primes <= 10000), so n^2 ~ 1.5M, which is very fast.

    states = [[] for _ in range(n)]
    states[0].append((0, 1, -1))

    best_length = 1
    best_end = 0
    best_prev_at_best_end = -1  # predecessor index for the best state at best_end

    for i in range(1, n):
        if elapsed() > DEADLINE:
            break

        pi = primes[i]
        gaps = pi - primes_arr[:i]  # vectorized: gap from primes[j] to primes[i] for all j < i

        # Collect all candidate (gap, length, prev) for state at i
        # Start with standalone: (0, 1, -1)
        candidates = [(0, 1, -1)]

        for j in range(i):
            sj = states[j]
            if not sj:
                continue

            gap = int(gaps[j])

            # Binary search in sj for rightmost entry with last_gap <= gap
            lo, hi = 0, len(sj)
            while lo < hi:
                mid = (lo + hi) // 2
                if sj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 0:
                last_gap_j, len_j, prev_j = sj[lo - 1]
                new_len = len_j + 1
                candidates.append((gap, new_len, j))

        # Filter to Pareto-optimal: sort by gap asc, keep only entries where
        # length is strictly greater than all previous entries
        if len(candidates) > 1:
            candidates.sort(key=lambda x: (x[0], -x[1]))
        filtered = []
        max_len = 0
        for gap, length, prev in candidates:
            if length > max_len:
                filtered.append((gap, length, prev))
                max_len = length
        states[i] = filtered

        if filtered:
            # The best length at i is the last entry (largest gap, largest length)
            last_len = filtered[-1][1]
            if last_len > best_length:
                best_length = last_len
                best_end = i
                best_prev_at_best_end = filtered[-1][2]

    # Reconstruct the sequence
    if best_length <= 1:
        return [primes[best_end]]

    # Follow the best state at best_end
    seq = []
    idx = best_end
    while idx != -1:
        seq.append(primes[idx])
        sj = states[idx]
        if not sj:
            break
        # Find the state entry that gives the best length (last entry)
        _, _, prev = sj[-1]
        idx = prev

    seq.reverse()

    # Verify the sequence is valid (nondecreasing gaps, all prime, increasing)
    def is_valid(seq):
        if not seq:
            return True
        for p in seq:
            if p > 10000:
                return False
        for k in range(len(seq) - 1):
            if seq[k] >= seq[k+1]:
                return False
        gaps = [seq[k+1] - seq[k] for k in range(len(seq) - 1)]
        for k in range(len(gaps) - 1):
            if gaps[k] > gaps[k+1]:
                return False
        return True

    if not is_valid(seq):
        # Fallback: try to reconstruct more carefully
        # The issue might be that sj[-1] doesn't correspond to the best_length path
        # Let's try a different reconstruction: track the full path
        # For now, fall back to greedy
        seq = [primes[0]]
        for p in primes[1:]:
            if len(seq) >= 2:
                last_gap = seq[-1] - seq[-2]
            else:
                last_gap = 0
            if p - seq[-1] >= last_gap:
                seq.append(p)

    # Try to extend the sequence greedily from the end
    if elapsed() < DEADLINE - 3:
        extended = list(seq)
        while elapsed() < DEADLINE - 3:
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
        
        if len(extended) > len(seq) and is_valid(extended):
            seq = extended

    # Try local search: try to replace elements to extend
    if elapsed() < DEADLINE - 5:
        # Try starting from different initial primes
        best_seq = list(seq)
        best_len = len(seq)
        
        # Try to find a longer sequence by modifying the end
        # For each position in the sequence, try to find a better continuation
        for start_idx in range(len(seq)):
            if elapsed() > DEADLINE - 5:
                break
            prefix = seq[:start_idx + 1]
            if len(prefix) >= 2:
                last_gap = prefix[-1] - prefix[-2]
            else:
                last_gap = 0
            last_prime = prefix[-1]
            
            # Greedy extend from here
            candidate = list(prefix)
            while elapsed() < DEADLINE - 5:
                if len(candidate) >= 2:
                    lg = candidate[-1] - candidate[-2]
                else:
                    lg = 0
                lp = candidate[-1]
                next_p = None
                for p in primes:
                    if p > lp and (p - lp) >= lg:
                        next_p = p
                        break
                if next_p is None:
                    break
                candidate.append(next_p)
            
            if len(candidate) > best_len and is_valid(candidate):
                best_len = len(candidate)
                best_seq = list(candidate)

        if len(best_seq) > len(seq):
            seq = best_seq

    return seq
