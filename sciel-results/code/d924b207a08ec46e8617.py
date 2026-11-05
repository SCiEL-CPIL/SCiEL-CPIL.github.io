import time
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 88  # seconds, safety margin before 90s wall-clock kill

    # ---------- helpers ----------
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        if n >= 0:
            is_prime[0] = 0
        if n >= 1:
            is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = 0
        return [p for p in range(2, n + 1) if is_prime[p]]

    primes = sieve(10000)
    n = len(primes)
    if n == 0:
        return []
    if n == 1:
        return primes

    # ---------- state representation ----------
    # For each index i we keep a Pareto front of states.
    # A state is (last_gap, length, prev_idx).
    # The front is sorted by last_gap ascending; within the same gap we keep
    # only the maximum length. Lengths are strictly increasing along the front.
    # A trivial state (sequence of length 1) has last_gap = 0, prev_idx = -1.

    fronts = [None] * n
    fronts[0] = [(0, 1, -1)]  # gap=0, len=1, prev=-1

    best_len = 1
    best_end = 0

    # Precompute index of each prime for O(1) gap lookups (optional, but cheap)
    # We'll compute gaps on the fly.

    # ---------- main DP ----------
    for i in range(1, n):
        if time.time() - start_time > DEADLINE:
            break

        pi = primes[i]
        # Collect candidate states for i.
        # Start with the trivial state.
        candidates = [(0, 1, -1)]

        # For each j < i, look at fronts[j] and try to extend states with last_gap <= (pi - primes[j]).
        for j in range(i):
            if time.time() - start_time > DEADLINE:
                break

            front_j = fronts[j]
            if front_j is None or len(front_j) == 0:
                continue

            gap = pi - primes[j]
            # Binary search in front_j for the largest entry with last_gap <= gap.
            # front_j is sorted by last_gap ascending.
            lo, hi = 0, len(front_j)
            while lo < hi:
                mid = (lo + hi) // 2
                if front_j[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is the number of entries with last_gap <= gap, so the best is at index lo-1.
            if lo > 0:
                last_gap, length, prev_idx = front_j[lo - 1]
                new_length = length + 1
                # New state: (gap, new_length, j)
                candidates.append((gap, new_length, j))

        # Now build the Pareto front for i from candidates.
        # Sort by gap ascending, then by length descending (so that for the same gap, the longest comes first).
        candidates.sort(key=lambda x: (x[0], -x[1]))

        # Sweep: keep only states where length is strictly greater than all previous kept lengths.
        front_i = []
        max_len_so_far = 0
        for gap, length, prev_idx in candidates:
            if length > max_len_so_far:
                front_i.append((gap, length, prev_idx))
                max_len_so_far = length

        fronts[i] = front_i

        # Update global best.
        if front_i:
            last_state = front_i[-1]
            if last_state[1] > best_len:
                best_len = last_state[1]
                best_end = i

    # ---------- reconstruct ----------
    if best_len <= 1:
        return [primes[best_end]]

    # We need to find the exact state at best_end that has length best_len.
    # In the front, lengths are strictly increasing, so there is at most one state
    # with the maximum length. If best_len is the max length in fronts[best_end], it's the last element.
    # But best_len is the global max, and best_end is the index where we first achieved it.
    # It's possible that fronts[best_end] has a state with length best_len. Let's find it.
    target_front = fronts[best_end]
    if not target_front:
        return [primes[best_end]]

    # Find the state in target_front with length == best_len.
    # Since lengths are strictly increasing, we can binary search or just scan.
    prev_idx = -1
    for gap, length, p_idx in target_front:
        if length == best_len:
            prev_idx = p_idx
            break

    if prev_idx == -1:
        # Fallback: use the last state
        prev_idx = target_front[-1][2]

    seq = [primes[best_end]]
    idx = best_end
    current_len = best_len
    while current_len > 1:
        front_at_idx = fronts[idx]
        if not front_at_idx:
            break
        # Find the state with length == current_len
        found = False
        for gap, length, p_idx in front_at_idx:
            if length == current_len:
                seq.append(primes[p_idx])
                idx = p_idx
                current_len -= 1
                found = True
                break
        if not found:
            # Should not happen; break to avoid infinite loop
            break

    seq.reverse()

    # ---------- verification ----------
    # Verify that seq is a valid increasing sequence of primes <= 10000
    # with nondecreasing gaps.
    if not seq:
        return [primes[0]]

    is_valid = True
    for k in range(len(seq) - 1):
        if seq[k] >= seq[k + 1]:
            is_valid = False
            break
        if seq[k] > 10000 or seq[k] < 2:
            is_valid = False
            break

    for k in range(len(seq)):
        p = seq[k]
        if p < 2:
            is_valid = False
            break
        # Simple primality check
        if p > 1:
            for d in range(2, isqrt(p) + 1):
                if p % d == 0:
                    is_valid = False
                    break
        if not is_valid:
            break

    if is_valid:
        for k in range(2, len(seq)):
            gap_curr = seq[k] - seq[k - 1]
            gap_prev = seq[k - 1] - seq[k - 2]
            if gap_curr < gap_prev:
                is_valid = False
                break

    if not is_valid:
        # Fallback: return a simple valid sequence.
        # Try to find the longest valid prefix of the sequence we constructed.
        valid_prefix = []
        for p in seq:
            valid_prefix.append(p)
            if len(valid_prefix) >= 2:
                g = valid_prefix[-1] - valid_prefix[-2]
                if len(valid_prefix) >= 3:
                    gp = valid_prefix[-2] - valid_prefix[-3]
                    if g < gp:
                        valid_prefix.pop()
                        break
            if not is_prime_check(p):
                valid_prefix.pop()
                break
        if valid_prefix:
            return valid_prefix
        return [primes[0]]

    return seq


def is_prime_check(p):
    if p < 2:
        return False
    if p == 2:
        return True
    if p % 2 == 0:
        return False
    for d in range(3, isqrt(p) + 1, 2):
        if p % d == 0:
            return False
    return True
