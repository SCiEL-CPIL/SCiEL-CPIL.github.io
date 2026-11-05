"""Rewritten program: DP over primes with (len, last_gap, last_prime) state, then bounded local search."""
# EVOLVE-BLOCK-START
import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, 101):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)

    def is_valid(seq):
        if len(seq) < 2:
            return True
        prev_gap = 0
        for i in range(1, len(seq)):
            gap = seq[i] - seq[i - 1]
            if gap < prev_gap:
                return False
            prev_gap = gap
        return True

    # DP: dp[i] = (length, last_gap) for the best sequence ending at prime i
    dp_len = [1] * n
    dp_gap = [0] * n
    dp_prev = [-1] * n

    for i in range(n):
        best_len = 1
        best_gap = 0
        best_j = -1
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            if gap >= dp_gap[j]:
                new_len = dp_len[j] + 1
                if new_len > best_len or (new_len == best_len and gap < best_gap):
                    best_len = new_len
                    best_gap = gap
                    best_j = j
        dp_len[i] = best_len
        dp_gap[i] = best_gap
        dp_prev[i] = best_j
        if time.time() > deadline:
            break

    # Collect all endpoints that achieve max_len, sorted by last_gap ascending
    max_len = 0
    for i in range(n):
        if dp_len[i] > max_len:
            max_len = dp_len[i]

    candidates = []
    for i in range(n):
        if dp_len[i] == max_len:
            candidates.append((dp_gap[i], i))
    candidates.sort()

    best_seq = []
    for _, end_idx in candidates:
        seq = []
        curr = end_idx
        while curr != -1:
            seq.append(primes[curr])
            curr = dp_prev[curr]
        seq.reverse()
        if len(seq) > len(best_seq):
            best_seq = seq[:]
        if time.time() > deadline:
            break

    if not best_seq:
        best_seq = [2, 3]

    # Local search: try inserting primes at all valid positions
    current_seq = best_seq[:]
    in_seq = set(current_seq)

    def try_insert_at(seq, new_p, pos):
        if pos > 0 and new_p <= seq[pos - 1]:
            return None
        if pos < len(seq) and new_p >= seq[pos]:
            return None
        new_seq = seq[:pos] + [new_p] + seq[pos:]
        if is_valid(new_seq):
            return new_seq
        return None

    no_improve = 0
    max_no_improve = 10000
    while time.time() < deadline and no_improve < max_no_improve:
        p_idx = random.randint(0, n - 1)
        p = primes[p_idx]
        if p in in_seq:
            continue

        pos = bisect.bisect_left(current_seq, p)
        found = False
        # Try positions around the insertion point
        for candidate_pos in range(max(0, pos - 2), min(len(current_seq) + 1, pos + 3)):
            new_seq = try_insert_at(current_seq, p, candidate_pos)
            if new_seq is not None:
                current_seq = new_seq
                in_seq = set(current_seq)
                no_improve = 0
                if len(current_seq) > len(best_seq):
                    best_seq = current_seq[:]
                found = True
                break
        if not found:
            no_improve += 1

    # Final validation
    if not is_valid(best_seq):
        return [2, 3]

    return best_seq
# EVOLVE-BLOCK-END
