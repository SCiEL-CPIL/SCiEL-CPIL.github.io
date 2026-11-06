import time
import random
import sys

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # safety margin before 90s kill
    MAX_VAL = 256

    # ---------- Core data structure: difference-set representation ----------
    # For a sequence [a1, a2, ..., an], define prefix sums P[0]=0, P[i]=a1+...+ai.
    # Consecutive block sum from i+1 to j = P[j] - P[i].
    # All block sums must be distinct, i.e., all differences P[j]-P[i] (0<=i<j<=n) are distinct.
    # Equivalently: for any two pairs (i,j) and (k,l) with i<j, k<l, (i,j)!=(k,l),
    # we need P[j]-P[i] != P[l]-P[k].
    #
    # When we append a new element x to the sequence, the new prefix sum is P[n+1] = P[n] + x.
    # The new differences introduced are P[n+1] - P[i] for i=0..n.
    # These must not collide with any existing difference.
    #
    # We maintain the set of all current differences. When appending x,
    # new_diffs = {P[n+1] - P[i] for i in range(n+1)}.
    # Check: new_diffs must be disjoint from diff_set, and new_diffs must have no internal collisions
    # (but since P is strictly increasing, P[n+1]-P[i] > P[n+1]-P[j] for i<j, so no internal collisions).

    def can_append(prefix_sums, diff_set, x):
        """Check if x can be appended. prefix_sums is list [0, P1, ..., Pn].
        Returns (new_prefix_sums, new_diffs) if valid, else None."""
        last_sum = prefix_sums[-1]
        new_sum = last_sum + x
        if new_sum > 256 * 10:  # sanity
            return None
        new_diffs = []
        for ps in prefix_sums:
            d = new_sum - ps
            if d in diff_set:
                return None
            new_diffs.append(d)
        return (prefix_sums + [new_sum], new_diffs)

    # ---------- Reverse greedy: build from largest to smallest ----------
    # Key insight: if we build the sequence in reverse (choosing elements from large to small),
    # the constraint structure is different. Let's think about it differently.
    #
    # Actually, let's use a different structural approach:
    # Build the sequence by choosing elements, but use a smarter search strategy.
    #
    # Approach 1: Reverse-order greedy
    # Instead of building left-to-right, we can think of it as:
    # Choose n distinct values from [1,256] such that all consecutive block sums are distinct.
    # The number of consecutive blocks for a sequence of length n is n(n+1)/2.
    # These must all be distinct sums.
    #
    # Let's try a bidirectional / reverse approach:
    # Start with the largest possible element and work backwards.

    def reverse_greedy():
        """Build sequence by picking elements from high to low, maintaining validity."""
        # We'll build the sequence in reverse: pick last element, then second-to-last, etc.
        # A sequence [a1, a2, ..., an] has block sums.
        # If we fix an = x, then the block sums ending at an are:
        #   x, a_{n-1}+x, a_{n-2}+a_{n-1}+x, ..., sum(a1..an)
        # This is equivalent to the forward problem on the reversed sequence.
        # So reverse greedy is just forward greedy on the reversed problem.
        # That's the same as forward greedy. Not helpful directly.
        
        # Let me try a different structural idea:
        # Use the fact that we can choose ANY subset of [1,256] in increasing order.
        # The greedy picks the smallest available. What if we use a different heuristic?
        return None

    # ---------- Strategy: Multi-start with different heuristics ----------
    
    # Forward greedy with different starting points and step patterns
    def greedy_with_start(start_val, max_val=MAX_VAL):
        """Greedily extend from start_val."""
        seq = []
        prefix_sums = [0]
        diff_set = set()
        
        for x in range(start_val, max_val + 1):
            result = can_append(prefix_sums, diff_set, x)
            if result is not None:
                prefix_sums, new_diffs = result
                seq.append(x)
                diff_set.update(new_diffs)
        return seq

    # Try multiple starting values
    best_seq = []
    for start in range(1, 30):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        seq = greedy_with_start(start)
        if len(seq) > len(best_seq):
            best_seq = seq

    # ---------- Strategy: Backtracking with pruning ----------
    # The key structural change: use a more intelligent search that doesn't just
    # greedily pick the next smallest, but explores alternatives.
    
    def backtrack_search(prefix_sums, diff_set, current_seq, min_val, depth_limit=100):
        """Depth-first search with pruning. Returns best sequence found."""
        nonlocal best_seq
        if time.time() - start_time > TIME_LIMIT - 5:
            return
        
        n = len(current_seq)
        if n > len(best_seq):
            best_seq = list(current_seq)
        
        if n >= depth_limit:
            return
        
        # Upper bound: can we possibly beat best_seq?
        # Maximum additional elements we can add: at most (MAX_VAL - min_val) but that's loose.
        # Better: the number of remaining values is MAX_VAL - min_val + 1.
        if n + (MAX_VAL - min_val + 1) <= len(best_seq):
            return
        
        # Try each possible next value
        for x in range(min_val, MAX_VAL + 1):
            if time.time() - start_time > TIME_LIMIT - 5:
                return
            
            result = can_append(prefix_sums, diff_set, x)
            if result is not None:
                new_ps, new_diffs = result
                current_seq.append(x)
                diff_set.update(new_diffs)
                
                backtrack_search(new_ps, diff_set, current_seq, x + 1, depth_limit)
                
                diff_set.difference_update(new_diffs)
                current_seq.pop()

    # Run backtracking from a few starting points with limited depth
    for start in [1, 2, 3, 5, 10]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        # Initialize with just the first element
        initial_ps = [0, start]
        initial_diffs = {start}
        backtrack_search(initial_ps, initial_diffs, [start], start + 1, depth_limit=len(best_seq) + 5)

    # ---------- Strategy: Local improvement / hill climbing ----------
    # Try to improve the current best by replacing elements or inserting new ones.
    
    def compute_all_diffs(seq):
        """Compute prefix sums and all differences for a sequence."""
        ps = [0]
        for x in seq:
            ps.append(ps[-1] + x)
        diff_set = set()
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                diff_set.add(ps[j] - ps[i])
        return ps, diff_set

    def is_valid_sequence(seq):
        """Check if all consecutive block sums are distinct."""
        if not seq:
            return True
        if len(set(seq)) != len(seq):
            return False
        if any(seq[i] >= seq[i+1] for i in range(len(seq)-1)):
            return False
        if any(x < 1 or x > MAX_VAL for x in seq):
            return False
        ps, _ = compute_all_diffs(seq)
        diff_set = set()
        for i in range(len(ps)):
            for j in range(i+1, len(ps)):
                d = ps[j] - ps[i]
                if d in diff_set:
                    return False
                diff_set.add(d)
        return True

    def try_improve(seq, time_budget):
        """Try to improve sequence by replacing one element with another."""
        seq = list(seq)
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            n = len(seq)
            for idx in range(n):
                if time.time() - start_time > time_budget:
                    break
                lo = seq[idx-1] + 1 if idx > 0 else 1
                hi = seq[idx+1] - 1 if idx < n-1 else MAX_VAL
                for new_val in range(lo, hi+1):
                    if new_val == seq[idx]:
                        continue
                    cand = seq[:idx] + [new_val] + seq[idx+1:]
                    if is_valid_sequence(cand):
                        # Try to extend
                        extended = list(cand)
                        ps, diff_set = compute_all_diffs(extended)
                        for x in range(extended[-1]+1, MAX_VAL+1):
                            result = can_append(ps, diff_set, x)
                            if result is not None:
                                ps, new_diffs = result
                                extended.append(x)
                                diff_set.update(new_diffs)
                        if len(extended) > len(seq):
                            seq = extended
                            improved = True
                            break
                if improved:
                    break
        return seq

    if len(best_seq) > 0:
        best_seq = try_improve(best_seq, TIME_LIMIT - 3)

    # ---------- Final greedy extension ----------
    if best_seq:
        ps, diff_set = compute_all_diffs(best_seq)
        extended = list(best_seq)
        for x in range(best_seq[-1]+1, MAX_VAL+1):
            result = can_append(ps, diff_set, x)
            if result is not None:
                ps, new_diffs = result
                extended.append(x)
                diff_set.update(new_diffs)
        if len(extended) > len(best_seq):
            best_seq = extended

    # ---------- Validate and return ----------
    if not best_seq:
        # Fallback: simple greedy
        best_seq = greedy_with_start(1)
    
    # Final validation
    assert is_valid_sequence(best_seq), f"Invalid sequence: {best_seq}"
    assert all(1 <= v <= MAX_VAL for v in best_seq)
    assert all(best_seq[i] < best_seq[i+1] for i in range(len(best_seq)-1))
    
    return best_seq
