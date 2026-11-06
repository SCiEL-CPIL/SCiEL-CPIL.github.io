"""Rewritten program using a greedy Maximum Coverage approach with bitset-based conflict detection."""

def construct():
    # We need a Sidon set containing 0 with all differences 1..30 present.
    # Minimize the largest element.
    
    # A Sidon set (B_2 sequence) has the property that all pairwise sums are distinct,
    # equivalently, all positive differences are distinct.
    
    # Strategy: Greedy construction where we maintain a set S containing 0.
    # We track which differences in 1..30 are already covered.
    # We also track all differences currently in the set to ensure Sidon property.
    # 
    # For each candidate x not in S, we compute:
    #   - new_diffs: differences x - s for s in S where 1 <= x-s <= 30 (if x > s)
    #     and s - x for s in S where 1 <= s-x <= 30 (if x < s)
    #   - all_new_diffs: all |x - s| for s in S
    #   - conflict: whether any of all_new_diffs is already in the set of existing differences
    #
    # We select candidates in order of increasing x (to minimize diameter), and for each
    # candidate, check if adding it maintains the Sidon property.
    #
    # Actually, let's use a smarter approach:
    # - Maintain S as a sorted list
    # - Maintain a set of all current positive differences
    # - Maintain a bitmask of which of 1..30 are covered
    # - At each step, try candidates x from 1 upward
    # - For candidate x, compute all new differences |x - s| for s in S
    # - Check none of these are in the existing difference set (Sidon constraint)
    # - Count how many new differences in 1..30 this would cover
    # - Among all valid candidates, pick the one with smallest x that covers at least one new required diff
    # - If no candidate covers a new required diff, pick the smallest valid candidate anyway
    # - Repeat until all 30 differences are covered
    
    S = [0]
    existing_diffs = set()
    covered = set()  # subset of {1, ..., 30}
    
    # We'll iterate, adding elements one by one
    # To ensure termination and small diameter, we limit search to reasonable bounds
    
    max_x = 200  # search up to this value for candidates
    
    while len(covered) < 30:
        best_x = None
        best_new_covered = -1
        
        for x in range(1, max_x + 1):
            if x in S:
                continue
            
            # Compute all new differences
            new_diffs = []
            for s in S:
                d = abs(x - s)
                if d == 0:
                    continue
                new_diffs.append(d)
            
            # Check Sidon constraint: no new diff should be in existing_diffs
            valid = True
            for d in new_diffs:
                if d in existing_diffs:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Count new required differences covered
            new_covered = 0
            for d in new_diffs:
                if 1 <= d <= 30 and d not in covered:
                    new_covered += 1
            
            if new_covered > best_new_covered:
                best_new_covered = new_covered
                best_x = x
                # We could break early if we find a candidate that covers many,
                # but let's keep scanning for a slightly better one in a bounded range
                # Actually, to minimize diameter, prefer smallest x among those with max new_covered
                # So don't break; continue scanning but only update if strictly better coverage
                # or same coverage but smaller x (but we scan x in increasing order, so first found with max is smallest)
        
        if best_x is None:
            # No valid candidate in range; extend max_x
            max_x *= 2
            continue
        
        # Add best_x to S
        S.append(best_x)
        S.sort()
        
        # Update existing_diffs and covered
        for s in S[:-1]:  # S[:-1] is S before adding best_x
            d = abs(best_x - s)
            existing_diffs.add(d)
            if 1 <= d <= 30:
                covered.add(d)
        
        # Early termination if we've covered all
        if len(covered) >= 30:
            break
    
    # Verify the result
    S.sort()
    all_diffs = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            all_diffs.append(S[j] - S[i])
    
    # Check Sidon property
    if len(set(all_diffs)) != len(all_diffs):
        # Fallback: use the original greedy approach
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
    
    # Verify all differences 1..30 are present
    diff_set = set(all_diffs)
    for d in range(1, 31):
        if d not in diff_set:
            # Fallback
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
    
    return S
