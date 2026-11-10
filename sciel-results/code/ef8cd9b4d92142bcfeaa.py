"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _is_prime_sieve(n):
    """Return a boolean array where index i is True if i is prime."""
    if n < 2:
        return np.zeros(n + 1, dtype=bool)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return is_prime

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    N = 10000
    is_prime = _is_prime_sieve(N)
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n_primes = len(primes)
    
    # DP: dp[i][j] = maximum length of valid subsequence ending at primes[i] 
    # with last gap exactly j (or >= j, we'll use >= for nondecreasing)
    # Actually, let's think differently.
    # A subsequence p_{i1} < p_{i2} < ... < p_{ik} is valid if
    # p_{i2} - p_{i1} <= p_{i3} - p_{i2} <= ... <= p_{ik} - p_{i_{k-1}}
    # 
    # State: (last_prime_index, last_gap) -> max_length
    # Transition: from (i, gap) we can go to (j, new_gap) where j > i, 
    # primes[j] - primes[i] >= gap, and new_gap = primes[j] - primes[i]
    #
    # But the state space could be large. Let's use:
    # dp[i][g] = max length of subsequence ending at primes[i] with last gap = g
    # where g ranges from 1 to N
    #
    # Actually, let's be smarter. For each prime index i, we track the best 
    # (max length) for each possible last gap value.
    # 
    # dp[i] = dict mapping last_gap -> max_length
    # Base case: dp[i] = {0: 1} for all i (single element, no gap yet)
    # But we need at least 2 elements to have a gap.
    # 
    # Let's define: dp[i][g] = max length of valid subsequence ending at primes[i] 
    # where the last gap is exactly g.
    # 
    # Initialization: For each i, dp[i][g] = 1 for all g (degenerate case)
    # Actually, a single prime has no gap, so let's say dp[i] starts as empty 
    # and we build from pairs.
    #
    # Better approach: 
    # dp[i] = dictionary: last_gap -> max_length
    # For i in range(n_primes):
    #   For each j > i:
    #     gap = primes[j] - primes[i]
    #     # Can we extend a subsequence ending at i with last gap <= gap to end at j with gap?
    #     best_len = 1  # just [primes[j]]
    #     for prev_gap, prev_len in dp[i].items():
    #         if prev_gap <= gap:
    #             best_len = max(best_len, prev_len + 1)
    #     # Also, a subsequence of length 2: [primes[i], primes[j]] has gap = gap
    #     best_len = max(best_len, 2)
    #     dp[j][gap] = max(dp[j].get(gap, 0), best_len)
    #
    # This is O(n_primes^2 * avg_dict_size) which could be slow.
    # n_primes ≈ 1229, so n_primes^2 ≈ 1.5M. If avg_dict_size is ~100, that's 150M ops.
    # Might be too slow in Python. Let's optimize.
    
    # Alternative: dp[i][g] where g is the last gap, but we can use an array.
    # Max gap is N-2 = 9998. So dp is a 2D array of size n_primes x (N+1).
    # That's 1229 * 10001 ≈ 12.3M entries. Each is an int. Feasible with numpy.
    
    # dp[i][g] = max length of valid subsequence ending at primes[i] with last gap = g
    # Initialize to 0 (invalid)
    # Base: for each i, we can start a subsequence of length 1 at primes[i].
    # But length 1 has no gap. Let's handle it specially.
    # 
    # Actually, let's define dp[i][g] = max length of valid subsequence of length >= 2
    # ending at primes[i] with last gap = g.
    # Then we also need to track length 1 subsequences separately, or handle the 
    # transition from length 1 to length 2.
    
    # Let me try a cleaner formulation:
    # Let best[i][g] = maximum length of a valid prime subsequence ending at primes[i]
    # where the last gap is exactly g.
    # 
    # For a subsequence of length 1: no gap, so we don't store it in best.
    # For a subsequence of length >= 2: the last gap is well-defined.
    # 
    # Transition: To form a subsequence ending at primes[j] with last gap = gap_j,
    # where gap_j = primes[j] - primes[i] for some i < j:
    #   - If the subsequence is [primes[i], primes[j]], length = 2
    #   - If we extend a subsequence ending at primes[i] with last gap g_i <= gap_j,
    #     then length = best[i][g_i] + 1
    # 
    # So: best[j][gap_j] = max(2, max_{i < j, g_i <= gap_j} (best[i][g_i] + 1))
    # where gap_j = primes[j] - primes[i]
    
    # This is still O(n^2 * G) where G is the number of gap values to check.
    # We can optimize by precomputing for each i, the max best[i][g] for g <= some value.
    
    # Let's try the straightforward approach first and see if it's fast enough.
    # n_primes ≈ 1229, so pairs ≈ 1.5M. For each pair, we need to find max best[i][g] for g <= gap.
    # If we precompute prefix max for each i, it's O(G) per i, total O(n*G) ≈ 1229*10000 ≈ 12M.
    # Then for each pair, it's O(1). Total O(n^2) ≈ 1.5M. Should be fine.
    
    max_gap = N  # Max possible gap is N-2, but let's use N for safety
    # best[i][g] for i in range(n_primes), g in range(1, max_gap+1)
    # Use a 2D array: shape (n_primes, max_gap + 1)
    best = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    
    # prefix_max[i][g] = max(best[i][1], best[i][2], ..., best[i][g])
    # We'll compute this on the fly or precompute after each row is filled.
    
    # Process primes in order. For each j, compute best[j][gap] for all i < j.
    # But best[j][gap] depends on best[i][g] for i < j, which are already computed.
    
    # For each j from 0 to n_primes-1:
    #   For each i from 0 to j-1:
    #     gap = primes[j] - primes[i]
    #     if gap > 0 and gap <= max_gap:
    #       # Find max best[i][g] for g in [1, gap]
    #       # This is prefix_max[i][gap]
    #       candidate = prefix_max[i][gap] + 1
    #       if candidate < 2:
    #         candidate = 2
    #       best[j][gap] = max(best[j][gap], candidate)
    
    # We need prefix_max[i] for each i. Since we process j in increasing order,
    # when processing j, all i < j have their best[i] fully computed.
    # So we can compute prefix_max[i] once after best[i] is done.
    
    # But best[j] is being computed while processing j, so we can't compute prefix_max[j] until j is done.
    # That's fine: we only need prefix_max[i] for i < j.
    
    # Let's precompute prefix_max for each row as we go.
    prefix_max = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    
    for j in range(n_primes):
        if time.time() > deadline:
            break
        # Compute best[j][gap] for all i < j
        for i in range(j):
            gap = primes[j] - primes[i]
            if gap <= 0 or gap > max_gap:
                continue
            # Get prefix_max[i][gap] = max best[i][1..gap]
            pm = prefix_max[i, gap]
            candidate = pm + 1
            if candidate < 2:
                candidate = 2
            if candidate > best[j, gap]:
                best[j, gap] = candidate
        # Now compute prefix_max[j]
        if j < n_primes - 1:  # Only needed if there are future j's
            # prefix_max[j][g] = max(best[j][1], ..., best[j][g])
            prefix_max[j, :] = np.maximum.accumulate(best[j, :])
    
    # Find the maximum length
    if n_primes == 0:
        return []
    
    max_len = 0
    best_j = -1
    best_gap = -1
    for j in range(n_primes):
        row_max = np.max(best[j, :])
        if row_max > max_len:
            max_len = row_max
            best_j = j
            # Find the gap that achieves this
            best_gap = int(np.argmax(best[j, :]))
    
    if max_len < 2:
        # Return the single largest prime or just [2]
        return [primes[0]]
    
    # Reconstruct the subsequence
    # We need to backtrack. Store parent pointers.
    # parent[j][gap] = (i, prev_gap) that led to best[j][gap]
    
    # Let's redo with parent tracking.
    best = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    parent_i = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    parent_gap = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    prefix_max = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    
    for j in range(n_primes):
        if time.time() > deadline:
            break
        for i in range(j):
            gap = primes[j] - primes[i]
            if gap <= 0 or gap > max_gap:
                continue
            pm = prefix_max[i, gap]
            candidate = pm + 1
            if candidate < 2:
                candidate = 2
                # If candidate is 2, the parent is (i, 0) meaning we start from i with no prior gap
                # But we need to track this. Let's use a special convention.
            if candidate > best[j, gap]:
                best[j, gap] = candidate
                if pm > 0:
                    # We extended a subsequence ending at i with some gap <= gap
                    # We need to find which gap at i gave the max
                    # This is tricky. We need to store which gap was optimal.
                    # Let's add another array: argmax_gap[i][g] = argmax_{h<=g} best[i][h]
                    parent_i[j, gap] = i
                    # We need to know which prev_gap at i was used
                    # For now, let's just store i and we'll figure out prev_gap during backtracking
                else:
                    # Length 2: [primes[i], primes[j]]
                    parent_i[j, gap] = i
                    parent_gap[j, gap] = 0  # 0 means no previous gap (start of subsequence)
        
        if j < n_primes - 1:
            prefix_max[j, :] = np.maximum.accumulate(best[j, :])
    
    # This is getting complicated with parent tracking. Let me simplify.
    # Just find the max length and return a valid subsequence of that length.
    # For the final answer, we can do a simpler reconstruction.
    
    # Actually, let's just return the length as the score. But we need to return the actual subsequence.
    # Let me add proper parent tracking.
    
    # Redo with proper tracking
    best = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    parent_i = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    parent_gap = np.full((n_primes, max_gap + 1), -1, dtype=np.int32)
    prefix_max = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    # argmax_prev_gap[i][g] = the gap h <= g that maximizes best[i][h]
    argmax_prev_gap = np.zeros((n_primes, max_gap + 1), dtype=np.int32)
    
    for j in range(n_primes):
        if time.time() > deadline:
            break
        for i in range(j):
            gap = primes[j] - primes[i]
            if gap <= 0 or gap > max_gap:
                continue
            pm = prefix_max[i, gap]
            if pm > 0:
                candidate = pm + 1
                # Find which gap at i gave the max
                # We need argmax_prev_gap[i][gap]
                if candidate > best[j, gap]:
                    best[j, gap] = candidate
                    parent_i[j, gap] = i
                    parent_gap[j, gap] = int(argmax_prev_gap[i, gap])
            else:
                # Length 2
                if 2 > best[j, gap]:
                    best[j, gap] = 2
                    parent_i[j, gap] = i
                    parent_gap[j, gap] = 0
        if j < n_primes - 1:
            prefix_max[j, :] = np.maximum.accumulate(best[j, :])
            # Compute argmax_prev_gap[j]
            # argmax_prev_gap[j][g] = argmax_{h<=g} best[j][h]
            # We can compute this by keeping track of the running argmax
            running_max = 0
            running_arg = 0
            for g in range(1, max_gap + 1):
                if best[j, g] > running_max:
                    running_max = best[j, g]
                    running_arg = g
                argmax_prev_gap[j, g] = running_arg
    
    # Find the best (j, gap)
    max_len = 0
    best_j = -1
    best_gap = -1
    for j in range(n_primes):
        row_max = int(np.max(best[j, :]))
        if row_max > max_len:
            max_len = row_max
            best_j = j
            best_gap = int(np.argmax(best[j, :]))
    
    if max_len < 2 or best_j < 0:
        return [primes[0]] if primes else []
    
    # Reconstruct
    result = []
    j = best_j
    gap = best_gap
    while j >= 0:
        result.append(primes[j])
        pi = parent_i[j, gap]
        pg = parent_gap[j, gap]
        if pi < 0 or pg < 0:
            break
        j = pi
        gap = pg
        if gap == 0:
            result.append(primes[j])
            break
    
    result.reverse()
    
    # Verify the result is valid
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if any(gaps[i+1] < gaps[i] for i in range(len(gaps)-1)):
            # Invalid, return greedy result as fallback
            values = []
            for p in range(2, 10001):
                if is_prime[p]:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
            return values
    
    return result

# EVOLVE-BLOCK-END
