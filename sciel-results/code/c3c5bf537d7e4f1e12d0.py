"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Generate all primes <= 10000 using Sieve of Eratosthenes
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    # We want to find the longest subsequence p_{i1} < p_{i2} < ... < p_{ik}
    # such that gaps g_j = p_{i_{j+1}} - p_{i_j} are nondecreasing: g_1 <= g_2 <= ... <= g_{k-1}
    #
    # DP approach:
    # dp[i][g] = longest valid subsequence ending at prime i with last gap exactly g
    # But gaps can be large. Instead, let's think differently.
    #
    # Alternative: For each pair (i, j) with i < j, the gap is primes[j] - primes[i].
    # We want longest path where consecutive gaps are nondecreasing.
    #
    # Let f(j, g) = max length of valid subsequence ending at prime j where the last gap is g.
    # Transition: f(j, g) = 1 + max{ f(i, g') : i < j, g' <= g, primes[j] - primes[i] == g }
    #            = 1 + max{ f(i, g') : i < j, primes[j] - primes[i] == g, g' <= g }
    #
    # Since primes[j] - primes[i] == g means primes[i] = primes[j] - g, we can look up i.
    # So for each j and each possible gap g, we check if primes[j] - g is a prime (find its index i),
    # then f(j, g) = 1 + max{ f(i, g') : g' <= g }
    #
    # To make this efficient, for each prime index i, we maintain a data structure that allows
    # querying max f(i, g') for g' <= g. Since we process primes in order, when we're at prime j,
    # all i < j have been processed.
    #
    # However, the number of distinct gaps is O(n^2) in worst case, but for primes up to 10000,
    # n ~ 1229, and gaps are at most ~10000. Let's use an array-based approach.
    #
    # Better approach: For each prime index i, store best[i] = dict mapping gap -> max length ending at i with that gap.
    # And prefix_max[i][g] = max{ best[i][g'] for g' <= g }. But this is expensive.
    #
    # Let's use a different formulation. Since the number of primes is ~1229, and we need
    # nondecreasing gaps, we can do:
    #
    # dp[j] = list of (last_gap, max_length) pairs for subsequences ending at prime j,
    #         sorted by last_gap, with max_length being the max length achievable with that last_gap.
    #
    # Transition to extend from i to j: gap = primes[j] - primes[i]
    #   For each (g', L) in dp[i] where g' <= gap: we can extend to get (gap, L+1) at j.
    #   Also, the single element [primes[j]] has gap 0 (or no previous gap), length 1.
    #
    # To merge into dp[j], we take the max length for each gap value, and maintain the prefix max.
    #
    # This could be O(n^2 * avg_dp_size). With n~1229, this might be ~1.5M * some factor.
    # Let's optimize: for each j, we aggregate all extensions from all i < j.
    
    # dp[i] will be a list of (gap, length) where gap is the last gap, and length is the max
    # subsequence length ending at prime i with that last gap. We'll also maintain a running
    # prefix-max structure for efficient queries.
    
    # Actually, let's use: for each prime index i, dp[i] is a dict: gap -> max_length.
    # And we maintain for each i a sorted list of gaps with their prefix max lengths.
    
    # To speed up, let's precompute for each j, the list of (i, gap) for all i < j.
    # That's O(n^2) pairs total, which is ~750K for n=1229. Manageable.
    
    # Let's use a more efficient approach:
    # For each j from 0 to n-1:
    #   candidates = {}  # gap -> max_length for subsequences ending at j
    #   candidates[0] = 1  # single element, treat gap as 0
    #   for i in range(j):
    #       gap = primes[j] - primes[i]
    #       # Query dp[i] for max length with last_gap <= gap
    #       best_prev = query_prefix_max(i, gap)
    #       if best_prev > 0:
    #           if gap not in candidates or candidates[gap] < best_prev + 1:
    #               candidates[gap] = best_prev + 1
    #   # Now build dp[j] from candidates, keeping only Pareto-optimal entries
    #   # (for each gap, the max length; and we can prune dominated entries)
    
    # For efficiency, let's store for each i:
    #   sorted_gaps[i] = sorted list of (gap, prefix_max_length)
    # where prefix_max_length[g] = max{ dp[i][g'] for g' <= g }
    
    import bisect
    
    sorted_gaps = [None] * n  # sorted_gaps[i] = list of (gap, prefix_max_len)
    dp_lengths = [None] * n   # dp_lengths[i] = dict gap -> max_length (raw, before prefix)
    
    best_overall = 1
    best_seq = [primes[0]]
    
    for j in range(n):
        if time.time() > deadline:
            break
        candidates = {0: 1}  # gap 0 means no previous gap (single element)
        
        for i in range(j):
            gap = primes[j] - primes[i]
            # Query sorted_gaps[i] for max length with gap' <= gap
            sg = sorted_gaps[i]
            if sg is None or len(sg) == 0:
                continue
            # Find rightmost gap' <= gap
            gaps_list = [x[0] for x in sg]
            pos = bisect.bisect_right(gaps_list, gap)
            if pos > 0:
                best_prev = sg[pos - 1][1]
                if best_prev > 0:
                    new_len = best_prev + 1
                    if gap not in candidates or candidates[gap] < new_len:
                        candidates[gap] = new_len
        
        # Build dp_lengths[j] and sorted_gaps[j]
        if not candidates:
            candidates = {0: 1}
        
        dp_lengths[j] = candidates
        # Sort by gap, compute prefix max
        items = sorted(candidates.items())  # (gap, length)
        sg_list = []
        cur_max = 0
        for gap, length in items:
            cur_max = max(cur_max, length)
            sg_list.append((gap, cur_max))
        sorted_gaps[j] = sg_list
        
        # Track best
        max_len_j = max(candidates.values())
        if max_len_j > best_overall:
            best_overall = max_len_j
    
    # Now reconstruct the best sequence.
    # We need to trace back. Let's do a separate pass to find the actual sequence.
    # We'll find which j achieves the max length, then backtrack.
    
    # Re-find the best ending position
    best_j = -1
    best_len = 0
    best_gap_at_end = 0
    for j in range(n):
        if sorted_gaps[j]:
            max_len = sorted_gaps[j][-1][1]
            if max_len > best_len:
                best_len = max_len
                best_j = j
                # Find the gap that achieves this
                # The prefix max is achieved at some gap; we need to find which gap gives exactly this max
                # Actually, we need to find a gap g such that dp_lengths[j][g] == best_len
                # and then backtrack.
    
    # Backtrack
    if best_j >= 0:
        # Find the gap at best_j that gives the max length
        target_len = best_len
        # We need to find (gap, prev_j) such that dp_lengths[best_j][gap] == target_len
        # and the previous element is some i with primes[best_j] - primes[i] == gap
        # and dp_lengths[i] has a gap' <= gap with length target_len - 1
        
        seq = [primes[best_j]]
        cur_j = best_j
        cur_len = target_len
        cur_gap = None
        
        # Find the gap at cur_j that achieves cur_len
        # Look through dp_lengths[cur_j]
        for gap, length in sorted(dp_lengths[cur_j].items(), reverse=True):
            if length == cur_len:
                cur_gap = gap
                break
        if cur_gap is None:
            # Shouldn't happen
            cur_gap = max(dp_lengths[cur_j].items(), key=lambda x: x[1])[0]
        
        while cur_len > 1:
            if time.time() > deadline:
                break
            # Find i such that primes[cur_j] - primes[i] == cur_gap
            # and dp_lengths[i] has a gap' <= cur_gap with max length == cur_len - 1
            prev_prime = primes[cur_j] - cur_gap
            # Find index of prev_prime in primes
            # Use binary search
            lo, hi = 0, n - 1
            found = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if primes[mid] == prev_prime:
                    found = mid
                    break
                elif primes[mid] < prev_prime:
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if found < 0:
                break
            
            # Find the best gap' in dp_lengths[found] with gap' <= cur_gap and length == cur_len - 1
            prev_gap = None
            for gap2, length2 in sorted(dp_lengths[found].items()):
                if gap2 <= cur_gap and length2 == cur_len - 1:
                    prev_gap = gap2
                    break
            if prev_gap is None:
                # Try to find any gap' <= cur_gap with max length >= cur_len - 1
                # Use the sorted_gaps structure
                sg = sorted_gaps[found]
                gaps_list = [x[0] for x in sg]
                pos = bisect.bisect_right(gaps_list, cur_gap)
                if pos > 0:
                    # sg[pos-1][1] should be >= cur_len - 1
                    # Find the actual gap that achieves cur_len - 1
                    for gap2, length2 in sorted(dp_lengths[found].items()):
                        if gap2 <= cur_gap and length2 >= cur_len - 1:
                            prev_gap = gap2
                            break
                if prev_gap is None:
                    break
            
            seq.append(primes[found])
            cur_j = found
            cur_len -= 1
            cur_gap = prev_gap
        
        seq.reverse()
        # Verify
        if len(seq) >= 2:
            gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
            if all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
                best_seq = seq
            else:
                # Fallback to greedy
                values = []
                for p in primes:
                    if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                        values.append(p)
                best_seq = values
        else:
            best_seq = seq
    else:
        # Fallback
        best_seq = [primes[0]]
    
    # Final verification
    if len(best_seq) >= 2:
        gaps = [best_seq[i+1] - best_seq[i] for i in range(len(best_seq)-1)]
        if not all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
            # Fallback to greedy
            values = []
            for p in primes:
                if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                    values.append(p)
            best_seq = values
    
    return best_seq
# EVOLVE-BLOCK-END
