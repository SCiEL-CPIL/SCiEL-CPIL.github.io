import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Lower bound on the maximum entry (scope)
    # Total distinct differences needed: n_rows * (k+1)*k/2
    # Max difference in a row of scope M is M.
    # So sum of differences available in 1..M is M(M+1)/2.
    # We need M(M+1)/2 >= n_rows * (k+1)*k/2
    total_diffs_needed = n_rows * (k + 1) * k // 2
    # Solve M(M+1)/2 >= total_diffs_needed
    # M^2 + M - 2*total_diffs_needed >= 0
    # M >= (-1 + sqrt(1 + 8*total_diffs_needed)) / 2
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2.0)
    
    # Upper bound: Greedy construction
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(new_diffs)
            rows.append(row)
        return rows, max(max(r) for r in rows)

    initial_rows, initial_max = greedy_construct()
    best_rows = [r[:] for r in initial_rows]
    best_max = initial_max
    
    # If the lower bound is close to the greedy result, try to improve from there.
    # We will try to find a solution with max entry < best_max.
    
    # Strategy: 
    # 1. Define a search space for each row.
    # 2. Use a backtracking solver that assigns rows one by one.
    # 3. For each row, we don't pre-generate all candidates (too many).
    #    Instead, we use a randomized construction of a row given the 'used_diffs' set.
    #    However, to make backtracking effective, we need to be able to "undo" and try different rows.
    #    
    # Improved Strategy: "Row-by-Row Backtracking with Heuristic Row Generation"
    # 
    # We maintain a global set of used differences.
    # We assign rows sequentially.
    # For row i, we attempt to generate a valid Golomb ruler that avoids used_diffs.
    # If we fail, we backtrack to the previous row and try a different variation.
    #
    # To avoid infinite loops in row generation, we limit the search depth for a single row
    # and use randomization.
    
    def generate_row_avoiding(used_diffs, max_scope, attempts=20):
        """
        Try to generate a single Golomb row of length k+1 with max entry <= max_scope
        that does not use any difference in used_diffs.
        Returns the row if successful, None otherwise.
        """
        for _ in range(attempts):
            row = [0]
            success = True
            for _ in range(k):
                # Determine valid candidates for the next element
                # Candidates must be > row[-1] and <= max_scope
                # And the new differences {x - a for a in row} must not intersect used_diffs 
                # AND must not intersect current row's differences
                
                current_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        current_diffs.add(row[j] - row[i])
                
                forbidden = used_diffs | current_diffs
                
                candidates = []
                # We can restrict the search range to improve speed
                # Start from row[-1] + 1 up to max_scope
                # To add randomness, we can shuffle a subset or pick randomly
                # But checking all is O(M^2). M is ~100-200. k is small.
                # Let's just iterate, but with some randomization in start point?
                # No, we need to find ANY valid x.
                # A good heuristic: try smaller x first to keep scope low, 
                # but randomize the order slightly to escape local minima.
                
                # Actually, for backtracking to work well, we want to generate *different* rows.
                # So we randomize the choice of x.
                
                valid_xs = []
                for x in range(row[-1] + 1, max_scope + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        valid_xs.append(x)
                
                if not valid_xs:
                    success = False
                    break
                
                # Heuristic: Prefer smaller x to keep scope low, but add noise
                # Sort by x, but pick randomly from the first 10? Or just random?
                # If we always pick min, we get the "greedy" row for that context.
                # If we pick random, we explore more.
                # Let's pick randomly from valid_xs to diversify.
                x = random.choice(valid_xs)
                row.append(x)
                current_diffs.update({x - a for a in row[:-1]})
                
            if success:
                # Verify the row is strictly increasing and within bounds
                if all(row[i] < row[i+1] for i in range(len(row)-1)) and row[-1] <= max_scope:
                    return row
        return None

    def solve_csp(target_max):
        """
        Try to find n_rows rows with max entry <= target_max.
        Uses backtracking.
        """
        used_diffs = set()
        rows = []
        
        def backtrack(idx):
            nonlocal used_diffs, rows
            if time.time() > deadline:
                return False
            
            if idx == n_rows:
                return True
            
            # Try to generate a row for position idx
            # To improve diversity, we can try multiple generations
            # But generate_row_avoiding already does 20 attempts.
            # We can call it multiple times in the backtrack loop?
            # No, if it fails once, it might fail again.
            # Better: generate_row_avoiding returns None if it can't find one in 20 tries.
            # If it returns a row, we commit. If backtracking happens, we need a DIFFERENT row.
            # So we need a way to generate the "next" candidate row.
            
            # Simple approach: Just call generate_row_avoiding.
            # If it returns a row, use it.
            # If backtracking occurs, we call it again.
            # Since it uses random.choice, it will likely return a different row.
            
            row = generate_row_avoiding(used_diffs, target_max, attempts=10)
            if row is None:
                return False
            
            # Compute diffs of the new row
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    row_diffs.add(row[j] - row[i])
            
            # Check for conflicts (should be none due to generation, but double check)
            if row_diffs & used_diffs:
                return False
            
            # Commit
            used_diffs.update(row_diffs)
            rows.append(row)
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            rows.pop()
            used_diffs.difference_update(row_diffs)
            
            # Try again? 
            # The current implementation of generate_row_avoiding uses random choices.
            # So calling it again will likely yield a different row.
            # However, if the search space is tight, we might loop.
            # We can add a limit on retries per node.
            # For now, let's just try once per backtrack step. 
            # If we want more robustness, we can loop here.
            
            # Let's try up to 5 times at this node before giving up
            for _ in range(5):
                row = generate_row_avoiding(used_diffs, target_max, attempts=10)
                if row is None:
                    continue
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        row_diffs.add(row[j] - row[i])
                
                if row_diffs & used_diffs:
                    continue
                
                used_diffs.update(row_diffs)
                rows.append(row)
                
                if backtrack(idx + 1):
                    return True
                
                rows.pop()
                used_diffs.difference_update(row_diffs)
            
            return False
        
        return backtrack(0)

    # Main loop: Decrease target_max from best_max - 1 down to lower_bound
    # We will try each target_max for a limited time.
    
    current_target = best_max - 1
    
    while current_target >= lower_bound and time.time() < deadline:
        # Give each target a time slice
        time_slice_start = time.time()
        time_slice_limit = 5.0 # 5 seconds per target
        
        found = False
        # Try multiple restarts for this target
        while time.time() - time_slice_start < time_slice_limit and not found:
            # Reset state for this attempt
            # We need to reset the 'rows' and 'used_diffs' inside solve_csp
            # The current solve_csp uses closures. We need to make it restartable.
            # Let's modify solve_csp to be self-contained per call or use a class.
            # For simplicity, I'll restructure slightly: solve_csp will do its own backtracking.
            
            # Re-implementing solve_csp to be cleaner for multiple attempts
            def attempt_solve(target):
                used = set()
                rows = []
                
                def bt(idx):
                    if time.time() > deadline:
                        return False
                    if idx == n_rows:
                        return True
                    
                    # Generate candidate rows
                    # We try a few candidates at this level
                    candidates_found = 0
                    while candidates_found < 10: # Try up to 10 different rows at this level
                        row = generate_row_avoiding(used, target, attempts=5)
                        if row is None:
                            break
                        
                        row_diffs = set()
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                row_diffs.add(row[j] - row[i])
                        
                        if row_diffs & used:
                            continue
                        
                        used.update(row_diffs)
                        rows.append(row)
                        
                        if bt(idx + 1):
                            return True
                        
                        rows.pop()
                        used.difference_update(row_diffs)
                        candidates_found += 1
                        if time.time() > deadline:
                            return False
                    return False
                
                return bt(0)
            
            if attempt_solve(current_target):
                # Verify
                all_diffs = set()
                valid = True
                for r in rows:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                
                if valid:
                    best_rows = [r[:] for r in rows]
                    best_max = max(max(r) for r in rows)
                    found = True
                    # If we found a solution, we can try to improve it further
                    # by lowering the target.
                    # But the loop structure decreases current_target automatically.
                    # However, we should update current_target to best_max - 1 to be precise?
                    # The loop `while current_target >= lower_bound` handles the decrease.
                    # But if we found a solution with max 100, and current_target was 105,
                    # the next iteration will try 104. We skipped 100-103.
                    # This is fine, we will eventually hit 100 if we keep going down.
                    # But we can jump to best_max - 1 to save time.
                    current_target = best_max - 1
                    break
        
        if not found:
            current_target -= 1
            # If we've spent too long on a target, reduce time slice for next?
            # Or just continue.
        
    return best_rows

