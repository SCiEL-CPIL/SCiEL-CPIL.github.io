import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Target differences: 1..30
    # We represent the set of covered differences as an integer bitmask.
    # Bit i (0-indexed) corresponds to difference i+1.
    # So bit 0 is diff 1, bit 29 is diff 30.
    
    def check_solution(A):
        # Verify A is a valid Sidon set covering 1..30
        diffs = set()
        for i in range(len(A)):
            for j in range(i):
                d = A[i] - A[j]
                if d in diffs:
                    return False
                if 1 <= d <= 30:
                    diffs.add(d)
        # Check if all 1..30 are covered
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def dfs(A, covered_mask, max_val):
        """
        A: sorted list of elements
        covered_mask: bitmask of differences 1..30 covered so far
        max_val: current maximum element in A
        Returns a valid list A or None
        """
        if time.time() > deadline:
            return None
            
        # Check if goal reached
        # Full mask for 1..30 is (1 << 30) - 1
        if (covered_mask & ((1 << 30) - 1)) == (1 << 30) - 1:
            return A[:]
        
        # Pruning: if we have too many elements
        if len(A) > 32:
            return None
            
        # Pruning: check if it's possible to cover remaining differences
        missing_count = 30 - bin(covered_mask & ((1 << 30) - 1)).count("1")
        if missing_count == 0:
            return A[:]
            
        slots_left = 32 - len(A)
        if slots_left <= 0:
            return None
            
        # Max new diffs we can add with slots_left elements
        # Each new element x adds diffs x - a for a in A.
        # Also diffs between new elements.
        # Rough upper bound: slots_left * len(A) + slots_left * (slots_left - 1) // 2
        max_possible_new_diffs = slots_left * len(A) + slots_left * (slots_left - 1) // 2
        if missing_count > max_possible_new_diffs:
            return None

        # Determine candidate next elements
        # We must add elements <= max_limit (which is passed via closure or arg)
        # To minimize diameter, we should try small elements first?
        # Actually, we are searching for ANY set within [0, M].
        # We can try all x in [max(A)+1, M].
        # But to find a solution faster, maybe try x that cover missing diffs?
        
        # Optimization: Only try x that cover at least one missing difference
        # This is a strong heuristic.
        
        current_max = A[-1]
        
        # Generate candidates
        # We can restrict candidates to those that produce a new difference in 1..30
        # Or just iterate x from current_max + 1 to M
        # M is up to maybe 100?
        
        # Let's iterate x
        for x in range(current_max + 1, M_limit + 1):
            # Check Sidon property for new x
            valid = True
            new_diffs = []
            for a in A:
                d = x - a
                if d <= 0: continue
                # Check if d is already in covered_mask (for 1..30) or in other diffs
                # We need to check against ALL existing positive differences, not just 1..30
                # So we need to maintain a set of all differences?
                # For speed, we can maintain a set of all differences in A.
                # But passing a set is expensive.
                # Alternative: check if x - a is in the set of differences of A.
                # We can precompute diffs of A? No, A changes.
                
                # Let's maintain a list of all diffs in A.
                # Or, since len(A) <= 32, len(diffs) <= 496.
                # Checking d in set(diffs) is O(1) if we have a set.
                
                if d in all_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            # Update covered_mask
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
                    
            # Heuristic: if new_mask == covered_mask, we didn't cover any new target diff.
            # It might still be useful if it enables future diffs, but unlikely to be optimal.
            # We can skip if it doesn't add new target diffs?
            # Risky. Let's allow it but deprioritize?
            # Actually, if we don't add a new target diff, we waste a slot.
            # Given tight constraints, let's only try x that adds at least one new target diff.
            if new_mask == covered_mask:
                continue

            # Recurse
            new_A = A + [x]
            new_all_diffs = all_diffs | set(new_diffs)
            res = dfs(new_A, new_mask, x, new_all_diffs)
            if res is not None:
                return res
                
        return None

    # Wrapper to manage all_diffs
    def dfs_wrapper(A, covered_mask, M_limit):
        # Precompute all diffs for initial A
        all_diffs = set()
        for i in range(len(A)):
            for j in range(i):
                all_diffs.add(A[i] - A[j])
        
        def inner_dfs(A, covered_mask, all_diffs):
            if time.time() > deadline:
                return None
                
            if (covered_mask & ((1 << 30) - 1)) == (1 << 30) - 1:
                return A[:]
                
            if len(A) > 32:
                return None
                
            missing_count = 30 - bin(covered_mask & ((1 << 30) - 1)).count("1")
            if missing_count == 0:
                return A[:]
                
            slots_left = 32 - len(A)
            if slots_left <= 0:
                return None
                
            max_possible_new_diffs = slots_left * len(A) + slots_left * (slots_left - 1) // 2
            if missing_count > max_possible_new_diffs:
                return None

            current_max = A[-1]
            
            # Try candidates x
            # To improve performance, we can try x that are "close" to existing elements
            # or just iterate.
            # Iterating up to M_limit.
            for x in range(current_max + 1, M_limit + 1):
                valid = True
                new_diffs = []
                for a in A:
                    d = x - a
                    if d in all_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                    
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                        
                if new_mask == covered_mask:
                    continue

                new_A = A + [x]
                new_all_diffs = all_diffs | set(new_diffs)
                
                res = inner_dfs(new_A, new_mask, new_all_diffs)
                if res is not None:
                    return res
                    
            return None
            
        return inner_dfs(A, covered_mask, all_diffs)

    # Main search loop
    # Start with M=30. If no solution, increase M.
    # We know a solution exists for some M.
    # Upper bound for M?
    # With 9 elements, max diff is roughly M.
    # If M=30, max diff is 30.
    # If M is larger, we have more flexibility.
    
    # Initial set: [0]
    # Covered mask: 0
    
    # To make search faster, we can add some initial elements.
    # But 0 is required.
    
    # Let's try M from 30 to 200.
    for M_limit in range(30, 201):
        # If time is running out, break
        if time.time() > deadline - 10:
            break
            
        res = dfs_wrapper([0], 0, M_limit)
        if res is not None:
            # Verify
            if check_solution(res):
                return res
                
    # Fallback: Greedy or Random
    # If DFS fails, return a valid set found by greedy (even if not optimal)
    # The seed program's greedy might work as a fallback.
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

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
    print(max(res))
