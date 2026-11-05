def construct():
    import time
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

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

    # DP: dp_len[i] = max length of valid subsequence ending at primes[i]
    # dp_prev[i] = index of predecessor that achieves dp_len[i]
    # dp_gap[i] = the gap (primes[i] - primes[dp_prev[i]]) used in the best subsequence
    #
    # For each i, we need: max over j < i of dp_len[j] + 1, subject to
    #   gap(j,i) = primes[i] - primes[j] >= last_gap_in_seq_j
    # where last_gap_in_seq_j is the gap between primes[dp_prev[j]] and primes[j]
    # (or 0 if dp_len[j] == 1).
    #
    # Key insight: for a fixed i, the constraint is primes[j] >= primes[i] - max_gap_j
    # where max_gap_j is the last gap in the best sequence ending at j.
    # But we want the MAXIMUM dp_len[j] among valid j.
    #
    # Since primes are sorted, for a given i, valid j are those with
    # primes[i] - primes[j] >= last_gap[j], i.e., primes[j] <= primes[i] - last_gap[j].
    # This is not simply a prefix because last_gap[j] varies.
    #
    # However, we can use the Pareto trick: for each possible last_gap value,
    # keep the max dp_len. But last_gap can be up to ~10000, so we can use an array.

    # last_gap[j] = gap between primes[prev] and primes[j] in the best seq ending at j
    # If dp_len[j] == 1, last_gap[j] = 0
    dp_len = np.ones(n, dtype=np.int32)
    dp_prev = np.full(n, -1, dtype=np.int32)
    last_gap = np.zeros(n, dtype=np.int32)  # gap into j from its predecessor

    # For efficient query: for a given i, find max dp_len[j] over j < i
    # where primes[j] <= primes[i] - last_gap[j]
    #
    # Alternative: iterate j and maintain a structure keyed by "effective threshold"
    # threshold[j] = primes[j] + last_gap[j]  (j is valid for i if threshold[j] <= primes[i])
    # We want max dp_len[j] among j < i with threshold[j] <= primes[i].
    #
    # As we process i from 0 to n-1, we can maintain a list of (threshold, dp_len)
    # for all j < i, and query max dp_len among those with threshold <= primes[i].
    #
    # Since primes[i] is increasing, and we add one j at a time, we can use a
    # Fenwick tree or segment tree over threshold values.
    # Threshold range: primes[j] + last_gap[j] <= 10000 + 10000 = 20000.

    MAX_THRESH = 20001  # primes[j] + last_gap[j] <= 10000 + ~10000

    # Fenwick tree for prefix max query
    class FenwickMax:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 1)
        
        def update(self, i, val):
            # 1-indexed
            i += 1
            while i <= self.n:
                if val > self.tree[i]:
                    self.tree[i] = val
                else:
                    # Can't decrease, but we only need max so this is fine
                    pass
                i += i & (-i)
        
        def query(self, i):
            # max over [0, i]
            i += 1
            res = 0
            while i > 0:
                if self.tree[i] > res:
                    res = self.tree[i]
                i -= i & (-i)
            return res

    # We need to track which j gives the max, not just the max value.
    # Use two Fenwick trees: one for max value, one for the argmax index.
    # Or: store (value, index) pairs.

    # Simpler approach: since n is only ~1229, O(n^2) with numpy vectorization
    # should be fast enough. Let's try a vectorized approach.

    # For each i, compute gaps to all j < i, then check constraint.
    # Constraint: gap >= last_gap[j] for each j.
    # We want max dp_len[j] among valid j.

    # Process in order, maintaining running best.
    # For each i:
    #   gaps = primes[i] - primes[:i]  (vector)
    #   valid = gaps >= last_gap[:i]
    #   Among valid j, find max dp_len[j], and the corresponding j.

    best_length = 1
    best_end = 0

    for i in range(1, n):
        if elapsed() > DEADLINE:
            break

        pi = primes[i]
        
        # Vectorized: gaps from all j < i to i
        gaps = pi - primes_arr[:i]
        
        # Valid predecessors: gap >= last_gap[j]
        valid_mask = gaps >= last_gap[:i]
        
        if not np.any(valid_mask):
            # No valid predecessor, dp_len[i] stays 1
            continue
        
        # Among valid, find max dp_len
        # Set invalid ones to -inf
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            continue
        
        valid_lengths = dp_len[valid_indices]
        max_idx_in_valid = np.argmax(valid_lengths)
        max_len = valid_lengths[max_idx_in_valid]
        best_j = valid_indices[max_idx_in_valid]
        
        new_len = max_len + 1
        dp_len[i] = new_len
        dp_prev[i] = best_j
        last_gap[i] = pi - primes[best_j]
        
        if new_len > best_length:
            best_length = new_len
            best_end = i

    # Reconstruct the sequence ending at best_end
    if best_length <= 1:
        return [primes[best_end]]

    seq = []
    idx = best_end
    while idx != -1:
        seq.append(primes[idx])
        idx = dp_prev[idx]

    seq.reverse()
    
    # Local search: try to improve by replacing elements
    # For each position, try nearby primes that maintain the gap constraint
    if elapsed() < DEADLINE - 10:
        # Multi-start local search
        best_seq = list(seq)
        
        # Try to extend from the end
        for _ in range(100):
            if elapsed() > DEADLINE - 10:
                break
            if len(best_seq) >= 2:
                last_gap_val = best_seq[-1] - best_seq[-2]
            else:
                last_gap_val = 0
            last_prime = best_seq[-1]
            # Find next prime > last_prime with gap >= last_gap_val
            next_p = None
            for p in primes:
                if p > last_prime and (p - last_prime) >= last_gap_val:
                    next_p = p
                    break
            if next_p is None:
                break
            best_seq.append(next_p)
        
        # Try replacing each element with a smaller valid prime to allow more room
        improved = True
        while improved and elapsed() < DEADLINE - 10:
            improved = False
            current = list(best_seq)
            for pos in range(len(current) - 1, 0, -1):
                if elapsed() > DEADLINE - 10:
                    break
                # Try to replace current[pos] with a smaller prime
                # Constraint: 
                #   If pos > 0: gap(pos-1, pos) = new_p - current[pos-1] >= gap(pos-2, pos-1) if pos >= 2
                #   If pos < len-1: gap(pos, pos+1) = current[pos+1] - new_p >= gap(pos-1, pos)
                
                lower_bound = 2
                if pos > 0:
                    if pos >= 2:
                        min_gap_before = current[pos-1] - current[pos-2]
                    else:
                        min_gap_before = 0
                    lower_bound = current[pos-1] + min_gap_before
                
                upper_bound = 10000
                if pos < len(current) - 1:
                    if pos > 0:
                        max_gap_after = current[pos+1] - current[pos]
                    else:
                        max_gap_after = 100000  # no constraint from before
                    upper_bound = current[pos+1] - max_gap_after
                
                # Find a prime in [lower_bound, upper_bound]
                # Prefer the smallest to leave room
                new_p = None
                for p in primes:
                    if p < lower_bound:
                        continue
                    if p > upper_bound:
                        break
                    # Check full constraint
                    valid = True
                    if pos > 0:
                        if pos >= 2:
                            if p - current[pos-1] < current[pos-1] - current[pos-2]:
                                valid = False
                        else:
                            pass  # no constraint
                    if pos < len(current) - 1:
                        if pos > 0:
                            if current[pos+1] - p < p - current[pos-1]:
                                valid = False
                        else:
                            pass
                    if valid:
                        new_p = p
                        break
                
                if new_p is not None and new_p != current[pos]:
                    current[pos] = new_p
                    improved = True
                    best_seq = current
                    break
        
        if len(best_seq) > len(seq):
            seq = best_seq
    
    return seq
