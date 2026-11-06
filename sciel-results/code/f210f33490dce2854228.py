"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Known perfect difference set for v=9, k=3 (covers 1..8? No, 0..8 diffs are 1,2,3,4,5,6,7,8? 
    # Diffs of {0,1,4,10,12,28,32,45,63}:
    # 1-0=1, 4-0=4, 10-0=10, 12-0=12, 28-0=28, 32-0=32, 45-0=45, 63-0=63
    # 4-1=3, 10-1=9, 12-1=11, 28-1=27, 32-1=31, 45-1=44, 63-1=62
    # 10-4=6, 12-4=8, 28-4=24, 32-4=28 (dup! 28-0=28 and 32-4=28? Wait. 32-4=28. 28-0=28. 
    # So {0,1,4,10,12,28,32,45,63} is NOT a Sidon set?
    # Let's check: 28-0=28, 32-4=28. Yes, collision.
    # The inspiration mentioned {0,1,4,10,12,28,32,45,63} as a "known perfect difference set".
    # A perfect difference set of order k has k(k-1) distinct differences.
    # For k=9, 9*8=72 diffs. Range 1..63 has 63 values. Impossible.
    # Maybe it was a typo in the inspiration or I am misremembering.
    # Let's use a verified small Sidon set that covers 1..30.
    # A greedy construction usually yields something around diameter 60-80.
    
    # We will implement a backtracking solver that finds a valid Sidon set covering 1..30.
    # Then we will try to minimize the diameter by backtracking with a tighter bound.
    
    # Strategy:
    # 1. Find ANY valid solution quickly to have a baseline.
    # 2. Try to find a solution with diameter D, decreasing D from a high estimate down to a lower bound.
    #    The lower bound for a Sidon set covering 1..n is roughly sqrt(n) * something?
    #    Actually, if we cover 1..30, we need at least 30 differences.
    #    A Sidon set of size k has k(k-1)/2 differences.
    #    To have 30 distinct small differences, we need k(k-1)/2 >= 30 -> k ~ 8 or 9.
    #    If k=9, max diff is M. The number of possible diffs is M.
    #    We need 1..30 to be in the diffs.
    #    Let's search for the minimum M.
    
    def is_valid_solution(arr):
        if len(arr) > 32: return False
        if 0 not in arr: return False
        if max(arr) > 10**6: return False
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def backtrack(arr, used_diffs, missing, max_val, target_diam):
        """
        arr: current sorted list
        used_diffs: set of differences already present
        missing: set of differences in 1..30 not yet covered
        max_val: current maximum element
        target_diam: maximum allowed diameter
        Returns: True if a valid set is found, and arr is updated.
        """
        if time.time() > deadline - 2:
            return False
        
        if not missing:
            # All 1..30 covered. Check Sidon property is maintained by used_diffs logic.
            # We must ensure no collisions in used_diffs. The logic below ensures that.
            return True

        # Heuristic: Pick the smallest missing difference? Or the one with fewest candidates?
        # Let's try to extend the set.
        # Candidate values for the next element x.
        # x must be > max_val.
        # x <= max_val + target_diam (actually x <= target_diam since min is 0)
        # Also x must be <= 10**6.
        
        # To speed up, we can restrict candidates.
        # If we add x, new diffs are {x - y for y in arr}.
        # These new diffs must not be in used_diffs.
        # Also, if any new diff is in missing, it reduces missing.
        
        # We can iterate x from max_val + 1 to target_diam.
        # Optimization: If we have time, try smaller x first (greedy) to keep diameter small.
        
        # Let's use a randomized order for x to escape local minima, but biased towards smaller x.
        # However, for exact backtracking, deterministic order is safer for finding *a* solution.
        # For minimizing diameter, we search in increasing order of target_diam.
        
        # Candidate generation:
        # We know we need to cover `missing`.
        # Let's try x in range(max_val + 1, target_diam + 1).
        
        # Optimization: 
        # If max_val + 1 > target_diam, return False.
        
        # Randomization: Shuffle candidates slightly? 
        # For now, deterministic ascending order.
        
        # To improve search, we can look ahead: 
        # Does x cover any missing diffs?
        # If not, it might be a "waste" of a slot, but might be necessary to enable future diffs.
        
        # Let's just iterate.
        # To avoid deep recursion if no solution, we can add a depth limit or timeout.
        
        # Heuristic: Try values that create new small diffs first.
        candidates = []
        for x in range(max_val + 1, target_diam + 1):
            # Check Sidon property with existing arr
            valid_x = True
            new_diffs = set()
            for y in arr:
                d = x - y
                if d in used_diffs:
                    valid_x = False
                    break
                new_diffs.add(d)
            
            if not valid_x:
                continue
            
            # Count how many missing diffs this x covers
            covered = sum(1 for d in new_diffs if d in missing)
            candidates.append((x, covered))
        
        # Sort candidates by coverage (descending), then by value (ascending)
        candidates.sort(key=lambda t: (-t[1], t[0]))
        
        for x, _ in candidates:
            # Calculate new state
            new_diffs = set()
            for y in arr:
                new_diffs.add(x - y)
            
            new_used = used_diffs | new_diffs
            new_missing = missing - new_diffs
            
            if backtrack(arr + [x], new_used, new_missing, x, target_diam):
                return True
        
        return False

    # Find the minimum diameter
    # Lower bound: 
    # We need 30 distinct differences.
    # If the set has size k, it has k(k-1)/2 differences.
    # The smallest difference is 1, largest is D.
    # It's known that for a Sidon set covering 1..n, the diameter is at least roughly 2*sqrt(n) or similar?
    # Actually, a perfect difference set of order 4 (size 4? No, size k has k(k-1) diffs? No, k(k-1)/2 positive diffs).
    # Let's start search from a reasonable lower bound.
    # Greedy usually finds ~60.
    # Let's try 40, 45, 50, etc.
    
    # First, find ANY solution to establish a valid upper bound.
    # Use a large target_diam to find a solution quickly.
    # Start with target_diam = 100.
    
    best_solution = None
    best_diam = float('inf')
    
    # Try to find a solution with diameter D, decreasing D.
    # We don't know the exact minimum, so we do a binary search or linear decrease.
    # Linear decrease from a known valid upper bound is safer.
    
    # Step 1: Find a valid solution with a loose bound (e.g., 100).
    # We use backtracking.
    # To make backtracking faster, we can use a "greedy" initial step to fill the set partially.
    
    # Let's implement a faster greedy to get an initial upper bound.
    def greedy_solve(max_diam):
        arr = [0]
        used_diffs = set()
        missing = set(range(1, 31))
        current_max = 0
        
        while missing and current_max < max_diam:
            found = False
            # Try x from current_max + 1 to max_diam
            # Prefer x that covers most missing
            best_x = -1
            best_covered = -1
            best_new_diffs = None
            
            for x in range(current_max + 1, max_diam + 1):
                valid = True
                new_diffs = set()
                for y in arr:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                covered = sum(1 for d in new_diffs if d in missing)
                if covered > best_covered:
                    best_covered = covered
                    best_x = x
                    best_new_diffs = new_diffs
            
            if best_x == -1:
                return None
            
            arr.append(best_x)
            used_diffs.update(best_new_diffs)
            missing -= best_new_diffs
            current_max = best_x
            
            if not missing:
                return arr
        
        return None
    
    # Get initial upper bound
    init_sol = greedy_solve(100)
    if init_sol:
        best_solution = init_sol
        best_diam = max(init_sol)
    
    # Now, try to improve the diameter.
    # Search for a solution with diameter < best_diam.
    # We can use backtracking with a tighter bound.
    
    # To speed up backtracking, we can use the greedy solution as a guide?
    # Or just run backtracking for D = best_diam - 1, best_diam - 2, ...
    
    # Backtracking might be slow if the space is large.
    # We will use a randomized backtracking with a time limit per D.
    
    def randomized_backtrack(arr, used_diffs, missing, max_val, target_diam, depth=0):
        if time.time() > deadline - 2:
            return False
        
        if not missing:
            return True
        
        if max_val >= target_diam:
            return False
        
        # Generate candidates
        candidates = []
        for x in range(max_val + 1, target_diam + 1):
            valid_x = True
            new_diffs = set()
            for y in arr:
                d = x - y
                if d in used_diffs:
                    valid_x = False
                    break
                new_diffs.add(d)
            
            if not valid_x:
                continue
            
            covered = sum(1 for d in new_diffs if d in missing)
            candidates.append((x, covered))
        
        if not candidates:
            return False
        
        # Shuffle candidates to randomize search path
        random.shuffle(candidates)
        # But bias towards high coverage? 
        # Let's sort by coverage desc, but add some noise.
        candidates.sort(key=lambda t: (-t[1], t[0]))
        
        for x, _ in candidates:
            new_diffs = set()
            for y in arr:
                new_diffs.add(x - y)
            
            new_used = used_diffs | new_diffs
            new_missing = missing - new_diffs
            
            if randomized_backtrack(arr + [x], new_used, new_missing, x, target_diam, depth + 1):
                return True
        
        return False

    # Try to find a better solution
    # We iterate D from best_diam - 1 down to a lower bound (e.g., 30)
    # For each D, we give it a small time slice.
    
    # To avoid getting stuck in one D for too long, we limit the time per D.
    # Total time budget for search: 100s.
    # If we try 20 values of D, we get 5s each.
    
    # Let's try D values: best_diam - 1, best_diam - 2, ...
    # If we find a solution, update best_diam and continue.
    
    # We can also use the fact that if we find a solution for D, we can try D-1.
    
    # Start D search
    current_target = best_diam - 1
    
    # Lower bound estimation:
    # A Sidon set of size k has k(k-1)/2 diffs.
    # To cover 1..30, we need at least 30 diffs.
    # k=9 -> 36 diffs.
    # The diameter of a Sidon set of size k is at least roughly k^2/4?
    # For k=9, 81/4 ~ 20.
    # Let's try down to 30.
    
    while current_target >= 30 and time.time() < deadline - 5:
        # Time slice for this D
        slice_deadline = time.time() + 5
        
        # Run randomized backtracking
        # We need to restore the time check inside the recursive function.
        # The recursive function uses `deadline` which is global to construct.
        # We can't easily change `deadline` inside the recursion without passing it.
        # Let's modify the function to take a deadline parameter or use a nonlocal variable.
        
        # For simplicity, let's just run it and hope the 5s slice is respected by the internal time check?
        # No, the internal check uses `deadline` (110s).
        # We need to pass a local deadline.
        
        def bt_with_deadline(arr, used_diffs, missing, max_val, target_diam, local_deadline):
            if time.time() > local_deadline:
                return False
            
            if not missing:
                return True
            
            if max_val >= target_diam:
                return False
            
            candidates = []
            for x in range(max_val + 1, target_diam + 1):
                valid_x = True
                new_diffs = set()
                for y in arr:
                    d = x - y
                    if d in used_diffs:
                        valid_x = False
                        break
                    new_diffs.add(d)
                
                if not valid_x:
                    continue
                
                covered = sum(1 for d in new_diffs if d in missing)
                candidates.append((x, covered))
            
            if not candidates:
                return False
            
            random.shuffle(candidates)
            candidates.sort(key=lambda t: (-t[1], t[0]))
            
            for x, _ in candidates:
                new_diffs = set()
                for y in arr:
                    new_diffs.add(x - y)
                
                new_used = used_diffs | new_diffs
                new_missing = missing - new_diffs
                
                if bt_with_deadline(arr + [x], new_used, new_missing, x, target_diam, local_deadline):
                    return True
            
            return False

        # Initialize state for backtracking
        init_arr = [0]
        init_used = set()
        init_missing = set(range(1, 31))
        
        if bt_with_deadline(init_arr, init_used, init_missing, 0, current_target, slice_deadline):
            # Found a solution with diameter <= current_target
            # We need to extract the solution. 
            # The function returns True/False, but doesn't populate the list in a way we can easily get.
            # We need to modify it to return the solution or populate a nonlocal variable.
            
            # Let's re-implement to return the list.
            pass # This approach is flawed. Let's use a class or nonlocal list.

    # Let's restart the search logic with a cleaner implementation that returns the solution.
    
    def find_solution_with_diam(target_diam, time_limit):
        """
        Uses randomized backtracking to find a Sidon set covering 1..30 with max element <= target_diam.
        Returns the set if found, else None.
        """
        start = time.time()
        end = start + time_limit
        
        # State: arr, used_diffs, missing
        # We use a recursive function that returns the solution list or None.
        
        def bt(arr, used_diffs, missing, max_val):
            if time.time() > end:
                return None
            
            if not missing:
                return arr[:]
            
            if max_val >= target_diam:
                return None
            
            # Generate candidates
            candidates = []
            for x in range(max_val + 1, target_diam + 1):
                valid_x = True
                new_diffs = set()
                for y in arr:
                    d = x - y
                    if d in used_diffs:
                        valid_x = False
                        break
                    new_diffs.add(d)
                
                if not valid_x:
                    continue
                
                covered = sum(1 for d in new_diffs if d in missing)
                candidates.append((x, covered))
            
            if not candidates:
                return None
            
            # Heuristic: Sort by coverage, then randomize slightly
            candidates.sort(key=lambda t: (-t[1], t[0]))
            
            # To avoid determinism issues, we can shuffle the top candidates
            # or just iterate.
            
            for x, _ in candidates:
                new_diffs = set()
                for y in arr:
                    new_diffs.add(x - y)
                
                new_used = used_diffs | new_diffs
                new_missing = missing - new_diffs
                
                res = bt(arr + [x], new_used, new_missing, x)
                if res:
                    return res
            
            return None

        return bt([0], set(), set(range(1, 31)), 0)

    # Main search loop
    # We already have best_solution from greedy.
    # Try to improve.
    
    # We can try target_diam values: best_diam - 1, best_diam - 2, ...
    # To be efficient, we can binary search or just linear decrease.
    # Linear decrease is safer for finding the *minimum*.
    
    # Let's try a range of target diameters.
    # If greedy found 60, try 59, 58...
    
    # To speed up, we can use the fact that if we fail for D, we might fail for D-1?
    # Not necessarily, but likely.
    
    # We will allocate time slices to each D.
    # Total time left: ~100s.
    # If we try 30 values, ~3s each.
    
    # Let's try D from best_diam - 1 down to max(30, best_diam - 20).
    
    # Also, we can try random restarts of the greedy to find a better initial upper bound?
    # Greedy is fast. Let's do 10 greedy restarts with different random seeds (if we add randomness).
    # Current greedy is deterministic. Let's add randomness to candidate selection.
    
    def greedy_solve_random(max_diam, seed=None):
        if seed is not None:
            random.seed(seed)
            
        arr = [0]
        used_diffs = set()
        missing = set(range(1, 31))
        current_max = 0
        
        while missing and current_max < max_diam:
            found = False
            best_x = -1
            best_covered = -1
            best_new_diffs = None
            
            # Randomize candidate order?
            # We can collect all valid candidates and pick the best one, or pick randomly from top-k.
            
            candidates = []
            for x in range(current_max + 1, max_diam + 1):
                valid = True
                new_diffs = set()
                for y in arr:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                covered = sum(1 for d in new_diffs if d in missing)
                candidates.append((x, covered, new_diffs))
            
            if not candidates:
                return None
            
            # Sort by coverage
            candidates.sort(key=lambda t: -t[1])
            
            # Pick the best, or randomly from top 3
            if len(candidates) > 3 and random.random() < 0.5:
                top_3 = candidates[:3]
                chosen = random.choice(top_3)
            else:
                chosen = candidates[0]
            
            best_x, _, best_new_diffs = chosen
            arr.append(best_x)
            used_diffs.update(best_new_diffs)
            missing -= best_new_diffs
            current_max = best_x
            
            if not missing:
                return arr
        
        return None

    # Run several greedy restarts to find a good initial upper bound
    for i in range(20):
        if time.time() > deadline - 50:
            break
        sol = greedy_solve_random(100, seed=i)
        if sol:
            diam = max(sol)
            if diam < best_diam:
                best_diam = diam
                best_solution = sol

    # Now, use backtracking to minimize further
    # Start from best_diam - 1 down to 30
    
    # We can use a "beam" of target diameters?
    # No, just iterate.
    
    # To save time, if we find a solution for D, we set best_diam = D and continue to D-1.
    # If we fail for D, we try D-1.
    
    # We allocate time based on remaining budget.
    
    current_target = best_diam - 1
    
    # If best_diam is already low (e.g., < 40), we might not have much room.
    
    while current_target >= 30 and time.time() < deadline - 10:
        # Allocate time slice
        # If we have 50s left and 20 targets, 2.5s each.
        remaining_targets = current_target - 30 + 1
        time_left = deadline - 10 - time.time()
        slice_time = max(0.5, time_left / remaining_targets)
        
        # Cap slice time to avoid spending too much on one
        slice_time = min(slice_time, 5.0)
        
        sol = find_solution_with_diam(current_target, slice_time)
        
        if sol:
            new_diam = max(sol)
            if new_diam < best_diam:
                best_diam = new_diam
                best_solution = sol
            # If found, we can continue to current_target - 1
            # But if new_diam < current_target, we might have skipped some?
            # No, we searched for max element <= current_target.
            # So new_diam <= current_target.
            # We should try current_target - 1 next.
        
        current_target -= 1
    
    if best_solution is None:
        # Emergency fallback
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            for y in a:
                d = x - y
                if d in diffs:
                    valid = False
                    break
            if valid:
                for y in a:
                    diffs.add(x - y)
                a.append(x)
            x += 1
        return a

    return best_solution
# EVOLVE-BLOCK-END
