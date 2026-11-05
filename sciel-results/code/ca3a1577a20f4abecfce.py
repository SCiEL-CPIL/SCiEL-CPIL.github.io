import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # Required differences: 1..30
    req_mask = 0
    for d in range(1, 31):
        req_mask |= (1 << d)

    best_solution = None
    best_M = 10**6

    def popcount(x):
        return bin(x).count('1')

    def solve_for_M(M, max_n=14):
        """
        Solve for a Sidon set in [0, M] covering 1..30.
        Uses a smarter DFS that only considers candidates x that create at least one new difference.
        """
        if time.time() > deadline:
            return None

        # We need to cover 30 differences.
        # A set of size n has n*(n-1)/2 differences.
        # For n=9, 36 diffs. For n=8, 28 diffs (not enough).
        # So n must be at least 9.
        # But wait, the differences don't have to be distinct?
        # "all positive differences distinct (a Sidon set)" -> Yes, they must be distinct.
        # So if n=8, max distinct differences is 28. We need 30. So n >= 9.
        # Actually, we just need 1..30 to be *in* the set of differences.
        # The set of differences can have more than 30 elements.
        # But if n=8, there are only 28 pairs, so at most 28 distinct differences.
        # So we cannot cover 30 distinct differences with n=8.
        # Thus n >= 9 is required.

        def dfs(current_set, covered_mask, missing_mask, depth):
            nonlocal best_solution, best_M
            if time.time() > deadline:
                return None

            if not missing_mask:
                sol = list(current_set)
                M_val = max(sol)
                if M_val < best_M:
                    best_M = M_val
                    best_solution = sol
                return sol

            # Pruning: If we have too many elements
            if depth >= max_n:
                return None

            # Heuristic: Check if remaining missing diffs can be covered.
            # Max new diffs from k more elements: k * len(current_set) + k*(k-1)/2.
            # This is an upper bound on new diffs.
            # We need to cover popcount(missing_mask) diffs.
            # If popcount(missing_mask) > max_possible_new_diffs, prune.
            # But this is loose.
            
            # Better pruning:
            # The smallest element in current_set is 0.
            # The largest is last.
            # Any new element x > last.
            # The new differences are x - y for y in current_set.
            # The smallest new difference is x - last.
            # If x - last > max_missing, then x cannot cover any missing diff > 0?
            # No, x - y could be small if y is large.
            # But x > last, so x - last >= 1.
            # If the smallest missing diff is d_min, and x - last > d_max_missing, 
            # then x cannot cover d_min?
            # Actually, x - y can be any value in {x - y : y in current_set}.
            # Since x > last, x - y > last - y >= 1.
            # The set of new diffs is {x - y | y in current_set}.
            # The smallest is x - last. The largest is x - 0 = x.
            # If x - last > max(missing), then x cannot cover any missing diff?
            # No, x - y could be <= max(missing).
            # Example: current=[0, 10], missing={1, 2, 3}. x=11. Diffs: 11, 1. 1 is missing.
            # So x - last = 1 <= 3.
            # If x - last > max_missing, then all new diffs are > max_missing.
            # So x is useless.
            # Thus, we only need to consider x such that x - last <= max_missing.
            # So x <= last + max_missing.
            
            last = current_set[-1]
            # Find max missing diff
            max_missing = 0
            temp_mask = missing_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                d = lsb.bit_length() - 1
                if d > max_missing:
                    max_missing = d
                temp_mask ^= lsb
            
            max_x = last + max_missing
            if max_x > M:
                max_x = M

            candidates = []
            
            # Optimization: If 1 is missing, we need x-y=1.
            # This implies some pair is consecutive.
            # We can check if 1 is missing and prioritize x = y+1 for some y.
            
            # Generate candidates:
            # x in [last+1, max_x]
            # For each x, check if it's valid (no diff collision) and compute gain.
            
            for x in range(last + 1, max_x + 1):
                valid = True
                new_diffs_mask = 0
                for y in current_set:
                    d = x - y
                    # d is always positive since x > y
                    if (covered_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if valid:
                    gain_mask = new_diffs_mask & missing_mask
                    gain = popcount(gain_mask)
                    if gain > 0:  # Only consider candidates that cover at least one missing diff
                        candidates.append((x, new_diffs_mask, gain))
            
            # Sort by gain descending, then x ascending
            candidates.sort(key=lambda t: (-t[2], t[0]))
            
            for x, new_diffs_mask, gain in candidates:
                next_covered = covered_mask | new_diffs_mask
                next_missing = missing_mask & ~new_diffs_mask
                
                res = dfs(current_set + [x], next_covered, next_missing, depth + 1)
                if res is not None:
                    return res
            
            return None

        # Start search
        # Initial set: [0]
        # Covered: 0
        # Missing: req_mask
        return dfs([0], 0, req_mask, 1)

    # Iterate M from lower bound
    # Lower bound for n=9 is 44.
    # Actually, for a Sidon set of size n in [0, M], we need n(n-1)/2 <= M(M+1)/2 roughly?
    # No, the differences are in 1..M. So we need n(n-1)/2 <= M.
    # For n=9, 36 <= M. So M >= 36.
    # But we also need to cover 1..30.
    # Let's start from 36.
    
    for M in range(36, 150):
        if time.time() > deadline:
            break
        
        sol = solve_for_M(M)
        if sol is not None:
            return sol
            
    # Fallback
    # If no solution found, return a known valid set (greedy)
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
