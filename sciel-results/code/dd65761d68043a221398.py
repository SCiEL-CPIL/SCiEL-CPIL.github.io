import time
import random

START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_sol = None
    best_max_val = float('inf')
    
    # Initial greedy to get a starting point / upper bound
    def greedy_construct():
        a = [0]
        used_diffs = set()
        missing = set(range(1, 31))
        for _ in range(32):
            if not missing:
                break
            best_x = None
            best_score = -1
            # Try adding x
            # Heuristic: x should be not too large, and cover many missing
            # Upper bound for x in greedy: maybe 2*max(a) + 30?
            # Let's just try x up to 200
            for x in range(max(a) + 1, 200):
                new_diffs = [x - y for y in a]
                # Check Sidon
                valid = True
                for d in new_diffs:
                    if d in used_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                # Check if x itself is already in a (it is, since x > max(a))
                # Count missing covered
                covered = sum(1 for d in new_diffs if d in missing)
                if covered > best_score:
                    best_score = covered
                    best_x = x
            if best_x is None:
                # If no good x found, break or fail
                break
            a.append(best_x)
            for y in a[:-1]:
                d = best_x - y
                used_diffs.add(d)
                if d in missing:
                    missing.remove(d)
        return a if not missing else None

    init_sol = greedy_construct()
    if init_sol:
        best_sol = init_sol
        best_max_val = max(init_sol)
        
    # Backtracking search
    def backtrack(current_set, used_diffs, missing, max_limit):
        nonlocal best_sol, best_max_val
        
        if time.time() - START_TIME > TIME_LIMIT:
            return None
            
        if not missing:
            # Found a solution
            if max(current_set) < best_max_val:
                best_max_val = max(current_set)
                best_sol = list(current_set)
            return list(current_set)
            
        if len(current_set) >= 32:
            return None
            
        # Pruning: Can we cover missing with remaining slots?
        remaining_slots = 32 - len(current_set)
        # Max new diffs from remaining_slots elements added to current_set
        # Each new element adds len(current_set) diffs.
        # Plus diffs among new elements.
        # Approx: remaining_slots * len(current_set) + remaining_slots*(remaining_slots-1)//2
        max_possible_new_diffs = remaining_slots * len(current_set) + remaining_slots * (remaining_slots - 1) // 2
        if max_possible_new_diffs < len(missing):
            return None
            
        # Generate candidates
        # x > max(current_set) and x <= max_limit
        start_x = max(current_set) + 1
        candidates = []
        for x in range(start_x, max_limit + 1):
            new_diffs = [x - y for y in current_set]
            # Check Sidon with existing
            valid = True
            for d in new_diffs:
                if d in used_diffs:
                    valid = False
                    break
            if not valid:
                continue
            
            # Score: number of missing diffs covered
            covered = 0
            for d in new_diffs:
                if d in missing:
                    covered += 1
            
            # Prefer higher covered, then smaller x
            candidates.append((covered, x))
            
        # Sort candidates: highest covered first, then smallest x
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for covered, x in candidates:
            if covered == 0:
                # If it doesn't cover any missing, it might be useless unless we are forced to?
                # But we can always add useless elements.
                # However, if we don't cover missing, we waste a slot.
                # Let's only try candidates that cover at least 1 missing if we have options?
                # Or just try them, but they are less likely to lead to success quickly.
                pass
            
            new_used_diffs = set(used_diffs)
            new_missing = set(missing)
            valid_add = True
            for y in current_set:
                d = x - y
                if d in new_used_diffs: # Should be caught above, but just in case
                    valid_add = False
                    break
                new_used_diffs.add(d)
                if d in new_missing:
                    new_missing.remove(d)
            
            if not valid_add:
                continue
                
            new_set = current_set + [x]
            res = backtrack(new_set, new_used_diffs, new_missing, max_limit)
            if res is not None:
                return res
                
        return None

    # Search for increasing max_limit
    # Start from a lower bound
    # Min diameter for Sidon set covering 1..30?
    # Try 40, 50, 60...
    # Or just start from 30 and go up.
    # To be safe, start from 30.
    
    current_limit = 30
    while time.time() - START_TIME < TIME_LIMIT:
        # If we have a solution, we can try to improve it by searching with tighter limit
        # But backtrack finds the first solution for a given limit.
        # We want the minimal max element.
        # So we search for limit = 30, 31, 32...
        # If we find one, we can stop?
        # No, we want to MINIMIZE the largest element.
        # So the first solution found for the smallest limit is the optimal (for that limit).
        # But is it possible that a solution with max element 40 exists but not 35?
        # Yes. So we must check 30, 31, ...
        
        # Optimization: If we found a solution with max_val V, we only need to check limits < V.
        # But we are increasing limit.
        # So if we start at 30 and go up, the first one we find is the minimal max element?
        # Not necessarily. A solution with max element 35 might require a specific structure that isn't found with limit 34,
        # but might be found with limit 35.
        # If we find a solution with limit L, it has max element <= L.
        # It might have max element < L.
        # We record best_sol.
        
        # To be efficient, we can just run backtrack with current_limit.
        # If it finds a solution, we update best_sol.
        # We can then try to see if a smaller max element is possible?
        # Actually, if we iterate L from 30 upwards, and for each L we find a valid set with max <= L,
        # the first L for which we find a set might not yield the absolute minimal max element
        # if the set found has max element L, but there exists a set with max element L-1 that we missed?
        # No, if we search limit L-1 first, we would have found it.
        # So iterating L from 30 upwards is correct.
        # However, backtrack might be slow.
        
        # Let's just run backtrack for current_limit.
        # To speed up, we can use the best_sol as a hint?
        # No, just run.
        
        res = backtrack([0], set(), set(range(1, 31)), current_limit)
        if res:
            # We found a solution with max <= current_limit
            # Since we are increasing current_limit, this is the minimal possible max element?
            # Yes, because we checked all smaller limits and found nothing.
            # Wait, did we check all?
            # If backtrack for limit 30 failed, it means no solution with max <= 30.
            # So if it succeeds for 31, the minimal max element is <= 31.
            # It could be 30? No, we checked 30.
            # So it is 31? Or maybe 30.5? No, integers.
            # So the minimal max element is current_limit?
            # Not necessarily. The solution found might have max element 31, but there could be a solution with max element 31 that is different.
            # But we just want the value.
            # So we can return this solution?
            # We should check if we can find a better one (smaller max) with more time?
            # No, if 30 failed, 31 is the best possible.
            # So we can return immediately?
            # Yes, assuming backtrack is complete (exhaustive).
            # It is exhaustive within the time limit.
            # If it returns None, it might be due to timeout.
            # So if it returns a solution, we are good.
            
            # One caveat: backtrack might time out and return None even if a solution exists.
            # So we should be careful.
            # But if it returns a solution, it's valid.
            
            # Let's verify the solution
            if is_valid(res):
                return res
            
        current_limit += 1
        
    # If we run out of time, return best found
    if best_sol:
        return best_sol
    else:
        # Fallback to greedy
        return greedy_construct()

def is_valid(a):
    if 0 not in a:
        return False
    if len(a) > 32:
        return False
    if max(a) > 10**6:
        return False
    diffs = []
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            d = a[j] - a[i]
            if d in diffs:
                return False
            diffs.append(d)
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

# Add a quick check for the initial greedy solution validity
# The greedy function above might return invalid if it breaks early?
# It checks `if not missing` at the end.
# But it might add elements that don't cover missing?
# It prioritizes covering missing.
# It should be valid.

# Let's add a global check in construct
