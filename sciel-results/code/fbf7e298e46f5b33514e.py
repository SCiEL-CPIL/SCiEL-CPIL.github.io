import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin under 60s kill
    n = 512
    target = 48

    def build_dp(arr):
        """Build subset-sum DP: dp[k] is bitset of sums of k-element subsets of arr."""
        m = len(arr)
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        return dp

    def is_valid_set(S):
        """Check if S is valid: no element is mean of any k-subset of others, k>=2."""
        m = len(S)
        if m <= 2:
            return True
        S_set = set(S)
        for i, x in enumerate(S):
            others = [v for j, v in enumerate(S) if j != i]
            m_others = len(others)
            dp = build_dp(others)
            for k in range(2, m_others + 1):
                target_sum = k * x
                if target_sum > m_others * n:
                    continue
                if (dp[k] >> target_sum) & 1:
                    return False
        return True

    def build_conflict_graph():
        """
        Build a conflict graph where an edge between a and b exists if there exists
        a third number c in [1,n] such that one of {a,b,c} is the mean of the other two.
        
        This captures 3-element arithmetic progressions: a, b, c where b = (a+c)/2
        (so b is the mean of a and c), or a = (b+c)/2, or c = (a+b)/2.
        
        Actually, for the mean condition: x is mean of subset {a,b} means x = (a+b)/2,
        so 2x = a+b. This means a, x, b form an arithmetic progression with x in the middle.
        
        So conflicts arise when three numbers form an arithmetic progression.
        """
        # For each pair (a, b), check if they form a 3-term AP with some c
        # a, b, c form AP if one is the mean of the other two
        # 2*middle = left + right
        
        # Build adjacency: conflict[a] = set of b such that a and b are in a 3-term AP
        # with some third number in [1,n]
        
        # A 3-term AP is determined by (start, diff): start, start+diff, start+2*diff
        # All must be in [1, n]
        
        conflicts = [set() for _ in range(n + 1)]
        
        # Enumerate all 3-term APs
        for diff in range(1, n // 2 + 1):
            for start in range(1, n - 2 * diff + 1):
                a = start
                b = start + diff
                c = start + 2 * diff
                # All three are in [1, n]
                # a, b, c form AP, b is mean of a and c
                # So b conflicts with a and c (b cannot be in set with both a and c)
                # Actually, the constraint is: no element is the mean of any subset of OTHER elements
                # So if we have {a, b, c} with b = (a+c)/2, then b is the mean of {a,c} which are other elements
                # This violates the condition.
                # So we cannot have all three in the set.
                # This means we need to model this as a hypergraph, not just a graph.
                # But for a greedy approach, we can use a graph where edges represent "potential conflict"
                
                # For greedy purposes: if we add b, we should avoid having both a and c
                # We can model this by saying a and c are "conflicting" in the presence of b
                # But that's a 3-way constraint.
                
                # Alternative: create edges between pairs that are "close" in AP sense
                # a and c are at distance 2*diff, with b in between
                # If we include b, we can't include both a and c
                # If we include a and c, we can't include b
                
                # For a simple graph model: connect a-c (the endpoints), since including both
                # endpoints makes it risky to include the middle
                conflicts[a].add(c)
                conflicts[c].add(a)
                # Also connect a-b and b-c? No, that would be too restrictive
                # The actual constraint is 3-way: {a,b,c} all together is bad
                # But {a,b} alone is fine, {b,c} alone is fine, {a,c} alone is fine
                # Only when all three are present is it bad
                
                # For greedy: when considering adding x, check if x creates a 3-term AP
                # with two already-selected numbers, or if adding x completes a 3-term AP
                # where x is the middle, or x is an endpoint
                
                pass
        
        return conflicts

    def can_add_to_set(x, current_set, n_val):
        """
        Check if x can be added to current_set without creating a violation.
        A violation occurs if x is the mean of some subset of current_set,
        or if some element in current_set is the mean of a subset that includes x.
        
        For efficiency, focus on small subsets first (2-element subsets are most common).
        """
        S = set(current_set)
        
        # Check if x is the mean of any 2-element subset of S
        # x = (a + b) / 2, so 2x = a + b
        # For each a in S, check if 2x - a is in S and != a
        for a in S:
            b = 2 * x - a
            if b != a and b in S:
                return False
        
        # Check if some element in S is the mean of {x, a} for a in S
        # For each a in S, check if (x + a) / 2 is in S and is an integer
        # (x + a) / 2 in S means x + a is even and (x+a)//2 in S
        if (x + 0) % 2 == 0:  # placeholder
            pass
        
        # More systematically: for each pair (x, a) where a in S, check if (x+a)/2 in S
        for a in S:
            s = x + a
            if s % 2 == 0:
                mid = s // 2
                if mid in S and mid != x and mid != a:
                    return False
        
        # For larger subsets, do a more expensive check
        # But for greedy, checking 2-element subsets catches most issues
        # Full validation will be done at the end
        
        # Check 3-element subsets where x is involved
        # x is mean of {a, b, c}: 3x = a+b+c
        # This is harder to check efficiently, so we'll rely on final validation
        
        return True

    def greedy_with_conflict_priority(order, current_best_len):
        """
        Greedily build a valid set, prioritizing numbers with fewer conflicts.
        """
        C = []
        C_set = set()
        for x in order:
            if len(C) >= target:
                break
            if x in C_set:
                continue
            if can_add_to_set(x, C, n):
                C.append(x)
                C_set.add(x)
                if len(C) > current_best_len:
                    current_best_len = len(C)
        return C

    # Build priority order based on conflict count
    # Numbers that are in fewer 3-term APs are "safer" to include
    def count_ap_involving(x):
        """Count how many 3-term APs involve x (as any position)."""
        count = 0
        # x as middle: a, x, c where x = (a+c)/2, so a = x-d, c = x+d
        for d in range(1, n + 1):
            a = x - d
            c = x + d
            if 1 <= a <= n and 1 <= c <= n:
                count += 1
        # x as first: x, x+d, x+2d
        for d in range(1, (n - x) // 2 + 1):
            if x + 2 * d <= n:
                count += 1
        # x as last: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if x - 2 * d >= 1:
                count += 1
        return count

    # Precompute conflict counts for all numbers
    conflict_counts = [0] * (n + 1)
    for x in range(1, n + 1):
        conflict_counts[x] = count_ap_involving(x)

    # Priority: numbers with fewer APs are safer
    # Sort by conflict count ascending, then by value
    priority_order = sorted(range(1, n + 1), key=lambda x: (conflict_counts[x], x))

    best = []
    rng = random.Random(42)

    # Try priority order
    C = greedy_with_conflict_priority(priority_order, len(best))
    if len(C) > len(best):
        best = C

    # Try reverse priority (more conflicts first, might leave room for others)
    reverse_priority = sorted(range(1, n + 1), key=lambda x: (-conflict_counts[x], x))
    C = greedy_with_conflict_priority(reverse_priority, len(best))
    if len(C) > len(best):
        best = C

    # Multiple random trials with priority-based ordering
    max_trials = 200
    for trial in range(max_trials):
        if time.time() > deadline - 2.0:
            break
        if len(best) >= target:
            break

        # Mix priority with randomization
        if trial % 4 == 0:
            order = list(priority_order)
            rng.shuffle(order[:len(order) // 2])  # Shuffle first half
        elif trial % 4 == 1:
            order = list(priority_order)
            # Weight toward lower conflict numbers
            weights = [1.0 / (1 + conflict_counts[x]) for x in order]
            # Simple weighted shuffle approximation
            order = [x for x in order if rng.random() < weights[order.index(x)] * 2]
            if len(order) < target:
                order = list(priority_order)
        elif trial % 4 == 2:
            # Odd numbers with priority
            odd_priority = [x for x in priority_order if x % 2 == 1]
            order = odd_priority
        else:
            # Random order
            order = list(range(1, n + 1))
            rng.shuffle(order)

        C = greedy_with_conflict_priority(order, len(best))
        if len(C) > len(best):
            best = C
            if len(best) >= target:
                break

    # Local improvement: try adding unused numbers
    if len(best) < target and time.time() < deadline - 3.0:
        best_set = set(best)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            candidates = [x for x in range(1, n + 1) if x not in best_set]
            # Sort by conflict count (fewer conflicts first)
            candidates.sort(key=lambda x: conflict_counts[x])
            for x in candidates:
                if len(best) >= target:
                    break
                if can_add_to_set(x, best, n):
                    # Full validation before committing
                    test_set = best + [x]
                    if is_valid_set(test_set):
                        best = test_set
                        best_set.add(x)
                        improved = True
                        break

    # Final validation
    if best and not is_valid_set(best):
        S = list(best)
        while S and not is_valid_set(S):
            S.pop()
        best = S

    return best
