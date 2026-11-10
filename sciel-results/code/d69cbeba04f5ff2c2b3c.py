import time
import random

DEADLINE = 0
TIME_LIMIT = 115

def construct(n_rows, k):
    global DEADLINE
    DEADLINE = time.time() + TIME_LIMIT
    
    # 1. Get a fast baseline using greedy
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return [[]]
    baseline_scope = max(max(row) for row in baseline_rows)
    
    # Theoretical lower bound
    # Total differences = n_rows * k*(k+1)/2
    # For D(7,5): 7 * 15 = 105
    # So smallest possible max difference is 105 (if we use 1..105 exactly)
    # But we need to fit them in rows of length k+1 starting at 0.
    # The max element in a row is at most the sum of some diffs, but usually close to the max diff used in that row.
    # Actually, the "scope" is the max entry in any row.
    # If we use diffs 1..105, the max diff is 105.
    # Can we have a row with max entry < 105? Yes, if the largest diff in that row is small.
    # But the global max entry must be at least the largest diff used anywhere?
    # No. If a row has elements [0, 2, 5], diffs are 2, 5, 3. Max entry 5. Max diff 5.
    # If a row has [0, 100], diff 100. Max entry 100.
    # So the max entry is at least the max difference used in any row.
    # Since all 105 differences 1..105 must be used, the max difference is 105.
    # Therefore, at least one row must contain a pair with difference 105.
    # If a row contains 0 and 105, its max entry is at least 105.
    # So the scope is at least 105.
    
    # Try to find a solution with scope = 105
    # If not found, try 106, etc.
    # But 105 is the counting bound. If a solution exists for 105, it's optimal.
    # If not, we want the next best.
    
    # The known best is 112.
    # Let's search from 105 upwards.
    
    best_rows = baseline_rows
    best_scope = baseline_scope
    
    # We can try to improve the baseline by searching for smaller scopes.
    # Start from 105.
    for M in range(105, best_scope + 1):
        if time.time() > DEADLINE:
            break
            
        # Try to construct a solution with max entry <= M
        # We use a randomized backtracking search
        # To speed up, we can use multiple attempts or a smarter heuristic
        
        # Limit attempts per M to avoid getting stuck
        found = False
        attempts = 500  # Adjust based on time
        start_time = time.time()
        
        # Pre-calculate available differences for this M?
        # The differences must be a subset of {1, ..., M}.
        # Since we need exactly 105 distinct differences, and M=105, we must use exactly {1..105}.
        # For M > 105, we can choose a subset of size 105 from {1..M}.
        
        if M == 105:
            # We must use all differences 1..105
            # This is a very tight constraint.
            # We can try to solve this specific instance.
            # It's equivalent to finding a permutation of {1..105} partitioned into 7 sets of 15,
            # such that each set can be the difference set of a row of 6 numbers.
            
            # A row [0, a1, a2, a3, a4, a5] has diffs:
            # a1, a2, a3, a4, a5
            # a2-a1, a3-a1, a4-a1, a5-a1
            # a3-a2, a4-a2, a5-a2
            # a4-a3, a5-a3
            # a5-a4
            
            # This is complex. Let's just use the general search.
            pass
        
        # Use randomized search with restarts
        while attempts > 0 and (time.time() - start_time) < (TIME_LIMIT * 0.8 / (best_scope - 104)):
            attempts -= 1
            rows = randomized_search(n_rows, k, M)
            if rows:
                # Verify
                if is_valid(rows, n_rows, k):
                    best_rows = rows
                    best_scope = M
                    found = True
                    break
            if time.time() > DEADLINE:
                return best_rows
        
        # If found a solution for this M, we can't do better (since we're increasing M from 105)
        # Wait, we are looking for the MINIMUM scope.
        # So if we find a solution for M=105, that's the best possible.
        # If we find M=106, that's the next best.
        # So we should stop as soon as we find a valid solution for the current M,
        # UNLESS we want to check if a smaller M works?
        # We are iterating M from 105 upwards.
        # So the first M for which we find a solution is the optimal scope (or close to it, depending on search completeness).
        # However, randomized search might fail to find a solution for M=105 even if one exists.
        # So we should try harder for smaller M, or try multiple M's in parallel?
        # Given the 120s limit, let's focus on the most likely optimal values.
        # The counting bound is 105.
        # If we find a solution for 105, return it.
        # If not, try 106, 107...
        
        if found:
            return best_rows

    return best_rows

def greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def is_valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if row[i] != row[i-1] + 1: # This check is wrong, they just need to be increasing
                pass
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def randomized_search(n_rows, k, M):
    """
    Try to find a set of n_rows rows, each with k+1 elements starting at 0,
    max element <= M, such that all differences are distinct.
    """
    used_diffs = set()
    rows = []
    
    # Order of rows to construct?
    # Maybe construct the "hardest" row first?
    # Or just random order.
    
    # We need to assign differences to rows.
    # Each row consumes k*(k+1)/2 differences.
    # Total diffs needed = n_rows * k*(k+1)/2
    # Available diffs = 1..M
    
    # Strategy: Backtracking with randomization.
    # Pick a row, fill it with random valid numbers.
    
    # To make it faster, we can pre-select the set of differences to use.
    # If M == n_rows * k*(k+1)/2, we must use all.
    # If M > ..., we can choose a subset.
    # Choosing a subset is hard.
    # Let's just pick numbers and check conflicts.
    
    # Heuristic: Start with the largest possible differences?
    # Or smallest?
    
    # Let's try building rows one by one.
    for _ in range(n_rows):
        row = [0]
        used_in_row = set()
        
        # Candidates for the next element
        # We want to pick x such that x - a_i are not in used_diffs and not in used_in_row
        # And x <= M
        
        # Randomized greedy with backtracking?
        # Simple random greedy might fail often.
        # Let's try a few iterations.
        
        success = False
        for attempt in range(100):
            current_row = [0]
            local_diffs = set()
            
            for _ in range(k):
                # Find valid candidates for next element
                # We can restrict the search space.
                # The next element x must be > current_row[-1] and <= M.
                # Also, the differences x - a_i must be valid.
                
                # Heuristic: Try larger x first?
                # Or random?
                
                # Let's collect valid candidates
                candidates = []
                # Range: current_row[-1] + 1 to M
                # But we can be smarter.
                # The maximum possible x is M.
                # The minimum x is current_row[-1] + 1.
                
                # To speed up, don't iterate all.
                # Just pick a random valid x?
                # How to check validity fast?
                # new_diffs = {x - a for a in current_row}
                # Check intersection with used_diffs and local_diffs.
                
                # Let's try a random sample of candidates.
                # If k=5, M=112.
                # Number of candidates is small enough to iterate if we prune?
                # 112 iterations per element, 5 elements, 7 rows. 112*5*7 = 3920 checks.
                # Each check is set intersection. Fast.
                
                # Let's just iterate from M downwards?
                # Or random?
                
                # Random search is better for finding *some* solution quickly.
                # Let's pick a random x from a valid range.
                
                # What is a good range?
                # If we pick x too small, we might run out of large diffs later?
                # If we pick x too large, we might not be able to fill the rest of the row?
                
                # Let's try picking x from [current_row[-1] + 1, M]
                # But filter by validity.
                
                # Optimization: Pre-compute which diffs are free?
                # free_diffs = set(range(1, M+1)) - used_diffs - local_diffs
                
                # For a given x, new_diffs = {x - a for a in current_row}
                # We need new_diffs subset of free_diffs.
                
                # Let's iterate candidates randomly.
                found_x = False
                # To avoid O(M) scan, we can shuffle a list of candidates?
                # M is small (<=112). O(M) is fine.
                
                # Let's try candidates in random order
                # Or try to find *any* valid x.
                
                # Let's collect all valid x first?
                # That's O(M * k).
                # Then pick one randomly?
                # This might lead to bias.
                
                # Let's just iterate 1..M and pick the first valid one?
                # That's deterministic greedy.
                # Let's add randomness.
                
                # Shuffle candidates?
                # Let's just pick a random valid candidate from a small window?
                
                # Actually, let's just iterate all x in [current_row[-1]+1, M]
                # and collect valid ones.
                valid_xs = []
                for x in range(current_row[-1] + 1, M + 1):
                    new_diffs = {x - a for a in current_row}
                    if not (new_diffs & used_diffs) and not (new_diffs & local_diffs):
                        valid_xs.append(x)
                        
                if not valid_xs:
                    break # Dead end for this attempt
                    
                # Pick a random valid x
                # Heuristic: Pick the one that leaves the most freedom?
                # Simple: Random
                x = random.choice(valid_xs)
                
                current_row.append(x)
                local_diffs |= {x - a for a in current_row[:-1]} # Update local_diffs
                
            if len(current_row) == k + 1:
                # Row complete
                # Commit
                rows.append(current_row)
                used_diffs |= local_diffs
                success = True
                break
                
        if not success:
            return None # Failed to construct this row
            
    return rows

# Main entry point
def construct(n_rows, k):
    global DEADLINE
    DEADLINE = time.time() + TIME_LIMIT
    
    # 1. Baseline
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return [[]]
    baseline_scope = max(max(row) for row in baseline_rows)
    
    # 2. Search for better solutions
    # Start from theoretical lower bound
    # Total diffs = n_rows * k * (k+1) // 2
    total_diffs = n_rows * k * (k+1) // 2
    min_scope = total_diffs
    
    best_rows = baseline_rows
    best_scope = baseline_scope
    
    # We want to find a solution with scope < best_scope
    # Try M from min_scope to best_scope - 1
    # If found, return it.
    
    # Time budget allocation
    # Give more time to smaller M (more likely to be optimal if it exists)
    # But if M=min_scope is very hard, we might waste time.
    
    # Let's try min_scope first with a large budget.
    # Then min_scope+1, etc.
    
    for M in range(min_scope, best_scope):
        if time.time() > DEADLINE:
            break
            
        # Determine time budget for this M
        # Remaining time = DEADLINE - time.time()
        # Let's spend a fraction of it.
        # For M=min_scope, spend 50% of remaining.
        # For next, 25%, etc.
        # Or just fixed number of attempts.
        
        # Let's use a fixed time slice per M, e.g., 10s
        # But if we have 100s left and 10 M values, that's fine.
        # If we have 100s left and 1 M value, we can use more.
        
        # Let's just try 1000 random searches per M, or until time slice expires.
        time_slice = 10.0
        start_t = time.time()
        found = False
        
        # Increase attempts if M is close to min_scope
        attempts = 2000 if M == min_scope else 500
        
        for _ in range(attempts):
            if time.time() - start_t > time_slice:
                break
            if time.time() > DEADLINE:
                break
                
            rows = randomized_search(n_rows, k, M)
            if rows:
                if is_valid(rows, n_rows, k):
                    best_rows = rows
                    best_scope = M
                    found = True
                    break
                    
        if found:
            # Found a solution for M.
            # Since we are iterating from min_scope upwards, this is the best possible
            # (within the range we search).
            # We can stop here?
            # Yes, because we want the minimum scope.
            # If we found one for M, we don't need to check M+1.
            # But wait, did we search M thoroughly?
            # If we used random search, we might have missed a solution for M.
            # But if we found one, it's valid.
            # The question is: is there a solution for M-1?
            # We already searched M-1 in the previous iteration (if M > min_scope).
            # So if we found a solution for M, it means we failed to find one for M-1.
            # This doesn't prove one doesn't exist for M-1, but it's our best effort.
            # Given the constraints, returning the first found valid solution in increasing order of M
            # is a reasonable heuristic.
            
            # However, we might want to continue searching for M-1 if we have time?
            # No, we already spent time on M-1.
            
            return best_rows

    return best_rows
