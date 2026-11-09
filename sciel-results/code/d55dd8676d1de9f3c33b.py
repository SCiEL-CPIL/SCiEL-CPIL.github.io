"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Sieve of Eratosthenes for primes up to 10000
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)

    # DP: dp[i] = max length of valid sequence ending at primes[i]
    # A valid sequence has nondecreasing gaps.
    # For a sequence ...p_j, p_i, the gap is primes[i] - primes[j].
    # We need: if the previous gap was g_prev, then primes[i] - primes[j] >= g_prev.
    # 
    # State: (ending_prime_index, last_gap) -> max_length
    # But last_gap can be large. Let's think differently.
    #
    # Actually, the constraint is on the sequence of gaps. If we have a sequence
    # p_{i1} < p_{i2} < ... < p_{ik} with gaps d_1, d_2, ..., d_{k-1} where
    # d_j = p_{i(j+1)} - p_{i(j)} and d_1 <= d_2 <= ... <= d_{k-1}.
    #
    # DP approach: dp[i][d] = max length of a valid sequence ending at primes[i] 
    # with last gap exactly d. But d can be up to ~10000, and n ~ 1229.
    # That's 1229 * 10000 ~ 12M states, too many.
    #
    # Alternative: dp[i] = for each ending prime i, store the best (max length) 
    # for each possible last gap. But we can use a different formulation.
    #
    # Key insight: For a fixed ending prime i, if we know the previous prime j,
    # the last gap is primes[i] - primes[j]. The constraint is that the gap before
    # that (call it g_prev) must be <= primes[i] - primes[j].
    #
    # So: dp[i] = max over all j < i of (dp_with_last_gap[j][primes[i]-primes[j]] + 1)
    # where dp_with_last_gap[j][g] = max length of valid sequence ending at j with last gap = g.
    #
    # This is still complex. Let's use a simpler DP:
    # dp[i] = max length of valid sequence ending at primes[i] (any last gap)
    # But we need to track the last gap to enforce the constraint.
    #
    # Let's use: best[i] = list of (gap, length) pairs, where for each possible last gap
    # ending at i, we store the max length. But this could be large.
    #
    # Better: For each i, we only need to consider transitions from j where
    # the gap primes[i]-primes[j] is >= the last gap in the sequence ending at j.
    #
    # Let's define: dp[i] = max length of valid sequence ending at primes[i].
    # And we also need to know, for each i, what last gaps are achievable with what lengths.
    #
    # Since n is only ~1229, we can do O(n^2) with careful tracking.
    #
    # Let dp[i] = max length of valid sequence ending at primes[i]
    # Let last_gap[i] = the last gap of the optimal sequence ending at i
    # But this greedy tracking doesn't work because a shorter sequence with a smaller
    # last gap might allow extending further.
    #
    # Correct approach: For each i, maintain a dictionary or array:
    #   seq_end[i][g] = max length of valid sequence ending at primes[i] with last gap = g
    # Transition: for j < i, g = primes[i] - primes[j], 
    #   seq_end[i][g] = max(seq_end[i][g], max_{g' <= g} seq_end[j][g'] + 1)
    #
    # To speed up the max over g' <= g, we can maintain prefix maxima for each j.
    #
    # n ~ 1229, max gap ~ 10000. O(n^2 * max_gap) is too slow.
    #
    # Alternative: For each j, precompute prefix_max[j][g] = max_{g' <= g} seq_end[j][g']
    # Then transition is O(1) per (i, j) pair. Total O(n^2).
    #
    # But storing prefix_max for each j: n * max_gap = 1229 * 10000 ~ 12M integers. OK.
    #
    # Let's implement this.

    max_gap = N  # max possible gap
    # seq_end[i] is an array of size max_gap+1, initialized to -inf (or 0 for impossible)
    # We'll use 0 to mean "no sequence ending here with this gap"
    # Actually, a single prime is a valid sequence of length 1 with no gap.
    # Let's handle base case: sequence of length 1, no gap.
    #
    # Let's redefine: 
    #   dp1[i] = 1 for all i (sequence of just primes[i])
    #   For sequences of length >= 2, we track the last gap.
    #
    # Let's use: 
    #   best_len[i] = max length of valid sequence ending at primes[i]
    #   And for transitions, we need: for each j < i, the max length of a valid sequence
    #   ending at j with last gap <= (primes[i] - primes[j]).
    #
    # Let prefix_max[j][g] = max length of valid sequence ending at j with last gap <= g.
    # For sequences of length 1, there's no last gap, so they can transition to any next prime.
    #
    # Let's define:
    #   seq_len[i][g] = max length of valid sequence ending at primes[i] with last gap exactly g
    #   For i, the base: a sequence of length 1 has no gap. Let's say last gap = 0 for length 1.
    #   Then seq_len[i][0] = 1 for all i.
    #   Transition: for j < i, g = primes[i] - primes[j] >= 1
    #     seq_len[i][g] = max(seq_len[i][g], max_{g' <= g} seq_len[j][g'] + 1)
    #
    # prefix_max[j][g] = max_{g' <= g} seq_len[j][g']
    #
    # We need prefix_max for all j. Computing it naively is O(max_gap) per j, total O(n * max_gap).
    # Then the DP transition is O(n^2).
    #
    # Total: O(n * max_gap + n^2) = O(1229 * 10000 + 1229^2) ~ 12M + 1.5M ~ 13.5M. Fine.

    INF = 0  # use 0 for impossible
    # seq_len[i] is a list of length max_gap+1
    # Initialize: seq_len[i][0] = 1 for all i
    seq_len = [[0] * (max_gap + 1) for _ in range(n)]
    for i in range(n):
        seq_len[i][0] = 1

    # prefix_max[j][g] = max(seq_len[j][0], ..., seq_len[j][g])
    # We'll compute this on the fly or precompute after filling seq_len[j].
    # But we need prefix_max[j] when processing i > j. So we compute prefix_max[j] 
    # after seq_len[j] is fully determined.
    #
    # Process i from 0 to n-1. For each i, first compute all transitions from j < i.
    # But we need prefix_max[j] for all j < i. So we maintain prefix_max as we go.
    #
    # Let's restructure:
    # For i in range(n):
    #   For j in range(i):
    #     g = primes[i] - primes[j]
    #     if g <= max_gap:
    #       best_prev = prefix_max[j][g]  # max seq_len[j][g'] for g' <= g
    #       if best_prev > 0:
    #         seq_len[i][g] = max(seq_len[i][g], best_prev + 1)
    #   # After processing all j, compute prefix_max[i]
    #   prefix_max[i] = [0] * (max_gap + 1)
    #   pm = 0
    #   for g in range(max_gap + 1):
    #     if seq_len[i][g] > pm:
    #       pm = seq_len[i][g]
    #     prefix_max[i][g] = pm

    prefix_max = []
    for i in range(n):
        for j in range(i):
            g = primes[i] - primes[j]
            if g <= max_gap:
                best_prev = prefix_max[j][g]
                if best_prev > 0:
                    if best_prev + 1 > seq_len[i][g]:
                        seq_len[i][g] = best_prev + 1
        # Compute prefix_max for i
        pm_arr = [0] * (max_gap + 1)
        pm = 0
        for g in range(max_gap + 1):
            if seq_len[i][g] > pm:
                pm = seq_len[i][g]
            pm_arr[g] = pm
        prefix_max.append(pm_arr)

        if time.time() > deadline:
            break

    # Find the best sequence
    best_len = 0
    best_end = -1
    best_gap = -1
    for i in range(n):
        for g in range(max_gap + 1):
            if seq_len[i][g] > best_len:
                best_len = seq_len[i][g]
                best_end = i
                best_gap = g

    # Backtrack to find the actual sequence
    if best_end == -1:
        return [2]

    # Reconstruct
    seq = [primes[best_end]]
    cur_i = best_end
    cur_g = best_gap
    while cur_g > 0:
        # Find j < cur_i such that primes[cur_i] - primes[j] == cur_g and
        # prefix_max[j][cur_g] == seq_len[cur_i][cur_g] - 1
        target_len = seq_len[cur_i][cur_g] - 1
        found = False
        for j in range(cur_i - 1, -1, -1):
            g = primes[cur_i] - primes[j]
            if g == cur_g:
                if prefix_max[j][g] >= target_len:
                    # Check if there's a gap g' <= g with seq_len[j][g'] == target_len
                    # Actually, we need the specific g' that achieved the max.
                    # This is tricky. Let's just pick the first j that works.
                    seq.append(primes[j])
                    cur_i = j
                    # We need to find the gap that was used at j.
                    # For simplicity, let's find g' such that seq_len[j][g'] == target_len
                    # and g' <= cur_g.
                    new_g = 0
                    for g2 in range(cur_g + 1):
                        if seq_len[j][g2] == target_len:
                            new_g = g2
                            break
                    cur_g = new_g
                    found = True
                    break
        if not found:
            break

    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
