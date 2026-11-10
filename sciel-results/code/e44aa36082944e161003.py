"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def primes_upto(n):
        """Sieve of Eratosthenes."""
        if n < 2:
            return []
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0] = 0
        if n >= 1:
            sieve[1] = 0
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n + 1) if sieve[i]]

    # Generate primes up to 10000
    primes = primes_upto(10000)
    m = len(primes)

    if m < 2:
        return primes

    # Fenwick Tree (BIT) over gap dimension for 2D dominance queries.
    # We process primes in order. For each prime j, we want to find the
    # longest nondecreasing-gap subsequence ending at j.
    #
    # State: for each prime j, we consider all i < j with gap d = primes[j] - primes[i].
    # We need: max over i < j of (best_len_ending_at_i_with_last_gap <= d) + 1.
    #
    # We maintain a Fenwick tree over the gap dimension (gaps range from 1 to 9998).
    # The tree stores, for each gap value g, the maximum sequence length of any
    # subsequence ending at some prime i < current j with last_gap exactly g.
    # Query(g) returns max length for last_gap <= g.
    #
    # To reconstruct, we track parent pointers.
    #
    # Key insight: when processing prime j, we query the BIT for each possible i < j.
    # But that's O(m^2) queries. Instead, we can do:
    #   For each j, for each i < j: d = primes[j] - primes[i]
    #   best = query(d)  # max length ending at some i' < j with last_gap <= d
    #   candidate_len = best + 1
    #   track the best candidate and its parent
    #
    # After processing j, we update the BIT at position d_j (the last gap of the best
    # subsequence ending at j) with the length.
    #
    # Wait - this isn't quite right. The BIT should store the max length for each
    # last-gap value across ALL primes processed so far. But the constraint is that
    # the previous prime i must be < j. Since we process in order, all i < j are
    # already in the BIT. Good.
    #
    # However, we need to be careful: the BIT stores max length for each last_gap.
    # When we query(d), we get the max length among all subsequences ending at any
    # prime i < j with last_gap <= d. This is exactly what we need.
    #
    # For reconstruction, we need to know WHICH prime i gave us the best length.
    # We can store, for each gap value g, the best (length, prime_index) pair.
    #
    # Let's use a Fenwick tree where each node stores (max_length, best_prime_index, best_prev_gap).
    #
    # Actually, let me think more carefully. The issue is that for a given prime j,
    # the best predecessor i might depend on the specific gap d = primes[j] - primes[i].
    # Different i values give different d values, so we need to query the BIT for each
    # distinct d.
    #
    # Algorithm:
    # 1. Initialize BIT with all zeros (no primes processed yet).
    # 2. Process prime 0: it's a base case, length 1, last_gap 0. Update BIT at gap 0.
    # 3. For j = 1, 2, ..., m-1:
    #    a. For each i in 0..j-1:
    #       d = primes[j] - primes[i]
    #       best_len, best_i, best_prev_gap = BIT.query(d)  # max length with last_gap <= d
    #       if best_len + 1 > current_best_for_j:
    #           current_best_for_j = best_len + 1
    #           best_pred = i
    #           best_pred_gap = best_prev_gap
    #    b. If no predecessor found, current_best_for_j = 1 (base case)
    #    c. Update BIT at position current_last_gap with (current_best_for_j, j, current_last_gap)
    #    d. Track global best.
    #
    # This is O(m^2 log(max_gap)) which is fine for m=1229.

    max_gap = primes[-1] - primes[0]  # ~9998
    bit_size = max_gap + 2  # 1-indexed, gaps are 1..max_gap, plus 0 for base case

    # Fenwick tree storing (max_length, prime_index, prev_gap)
    # We use 1-indexed: gap g maps to index g+1 (so gap 0 -> index 1, gap 1 -> index 2, etc.)
    bit_len = [0] * (bit_size + 2)
    bit_prime = [-1] * (bit_size + 2)
    bit_prev_gap = [0] * (bit_size + 2)

    def bit_update(idx, length, prime_idx, prev_gap):
        """Update BIT at position idx (1-indexed) with (length, prime_idx, prev_gap)."""
        while idx <= bit_size + 1:
            if length > bit_len[idx]:
                bit_len[idx] = length
                bit_prime[idx] = prime_idx
                bit_prev_gap[idx] = prev_gap
            idx += idx & (-idx)

    def bit_query(idx):
        """Query BIT for max (length, prime_idx, prev_gap) over indices 1..idx."""
        best_len = 0
        best_prime = -1
        best_pg = 0
        while idx > 0:
            if bit_len[idx] > best_len:
                best_len = bit_len[idx]
                best_prime = bit_prime[idx]
                best_pg = bit_prev_gap[idx]
            idx -= idx & (-idx)
        return best_len, best_prime, best_pg

    # dp_len[j] = best length of nondecreasing-gap subsequence ending at primes[j]
    # dp_parent[j] = (prev_prime_index, prev_gap) for reconstruction
    dp_len = [0] * m
    dp_parent = [(-1, 0)] * m

    best_global_len = 0
    best_global_end = 0

    for j in range(m):
        if j == 0:
            dp_len[j] = 1
            dp_parent[j] = (-1, 0)
            # Update BIT: gap 0, length 1, prime index 0
            bit_update(1, 1, 0, 0)
            best_global_len = 1
            best_global_end = 0
            continue

        best_len_j = 1
        best_pred = -1
        best_pred_gap = 0

        for i in range(j):
            d = primes[j] - primes[i]
            # Query BIT for max length with last_gap <= d
            q_len, q_prime, q_pg = bit_query(d + 1)  # gap d maps to index d+1
            if q_len + 1 > best_len_j:
                best_len_j = q_len + 1
                best_pred = i
                best_pred_gap = q_pg

        dp_len[j] = best_len_j
        dp_parent[j] = (best_pred, best_pred_gap)

        # Update BIT with the new prime j
        # The last gap of the subsequence ending at j is:
        #   if best_pred == -1: 0 (base case)
        #   else: primes[j] - primes[best_pred]
        if best_pred == -1:
            last_gap = 0
        else:
            last_gap = primes[j] - primes[best_pred]

        bit_update(last_gap + 1, best_len_j, j, last_gap)

        if best_len_j > best_global_len:
            best_global_len = best_len_j
            best_global_end = j

        # Deadline check
        if time.time() - t0 > DEADLINE:
            break

    # Reconstruct the sequence
    seq = [primes[best_global_end]]
    cur_idx = best_global_end

    while dp_parent[cur_idx][0] != -1:
        prev_i, prev_gap = dp_parent[cur_idx]
        seq.append(primes[prev_i])
        cur_idx = prev_i

    seq.reverse()

    # Verify the sequence has nondecreasing gaps
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        valid = True
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                valid = False
                break
        if not valid:
            # Fallback to greedy
            values = [primes[0]]
            for p in primes[1:]:
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
            return values

    return seq

# EVOLVE-BLOCK-END
