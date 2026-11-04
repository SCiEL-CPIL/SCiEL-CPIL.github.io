"""
Strategy: Iterated Local Search with Conflict-Driven Repair.
1. Fast Greedy Initialization: Build rows using a heuristic that balances
   difference usage to avoid early exhaustion of small differences.
2. Local Search: Identify the row with the largest entry. Attempt to reconstruct
   it using a constrained backtracking solver that must fit within the existing
   set of used differences (excluding those of the row being reconstructed).
3. Iteration: Repeat until time runs out or no improvement is possible.
"""
import random
import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Theoretical lower bound for max difference
    # If we use diffs 1..total_diffs, the max entry is roughly total_diffs
    # But we need to pack them into rows.
    
    best_rows = None
    best_max = float('inf')
    best_used_diffs = 0  # Bitset of used differences in best solution

    def get_time_left():
        return deadline - time.time()

    def greedy_build_row(used_diffs, max_val_hint, rng, k, existing_rows=None, target_row_idx=None):
        """
        Builds a single row of length k+1 starting with 0.
        'used_diffs' is a bitset of differences ALREADY USED by other rows.
        This row must not use any difference in 'used_diffs'.
        It also cannot use the same difference twice within itself.
        
        Heuristic: 
        - We want to keep values small.
        - We want to avoid differences that are 'popular' or likely to be needed elsewhere.
        - Since we are building one row at a time in the greedy phase, we just try to pack tight.
        - In the repair phase, used_diffs is fixed (from other rows), so we must fit into the 'holes'.
        """
        row = [0]
        current_max = 0
        # We need to pick k more numbers.
        # To make it fast, we estimate the required max value.
        # If we have U used diffs, and we need D new diffs, the max value must be at least 
        # the D-th smallest integer not in U.
        
        # Estimate a starting limit
        limit = max_val_hint if max_val_hint > 0 else total_diffs + 10
        
        # We'll try to build the row.
        # If we fail, we expand the limit.
        
        attempts = 0
        while True:
            attempts += 1
            if attempts > 50 or get_time_left() < 0.1:
                return None, limit
            
            row = [0]
            row_used_diffs = 0 # Bitset for diffs used in THIS row
            
            # We need to fill k positions.
            # To be efficient, we don't backtrack deeply here. We use a greedy-with-lookahead.
            # Actually, for repair, we need a proper backtracker because the space is constrained.
            # But for initial greedy, a simple forward check is enough.
            
            # Let's use a recursive helper for this row only.
            def build_row_recursive(col_idx, max_allowed):
                if col_idx == k + 1:
                    return row[:], row_used_diffs
                
                prev_val = row[-1]
                # Try candidates from prev_val + 1 up to max_allowed
                # Optimization: We can skip values that create a diff already in row_used_diffs or used_diffs
                
                # Generate candidates in random order? No, increasing order helps keep max small.
                # But to avoid getting stuck, we might want to try a few random ones if we fail?
                # For now, deterministic increasing order with a hard limit.
                
                # To speed up, we can jump to the next valid candidate.
                # A diff d is valid if (used_diffs >> d) & 1 == 0 and (row_used_diffs >> d) & 1 == 0
                
                # We iterate x from prev_val + 1 to max_allowed
                # This can be slow if max_allowed is large.
                # Optimization: Instead of iterating x, iterate potential diffs?
                # No, x determines the diffs.
                
                # Let's just iterate x, but break early if we find a solution.
                # If k is small (e.g., 7), this is fine.
                
                for x in range(prev_val + 1, max_allowed + 1):
                    if get_time_left() < 0.05:
                        return None, None
                    
                    valid = True
                    new_row_diffs = 0
                    # Check diffs against previous entries in this row
                    for i in range(col_idx):
                        d = x - row[i]
                        if d <= 0:
                            valid = False
                            break
                        if (used_diffs >> d) & 1:
                            valid = False
                            break
                        if (row_used_diffs >> d) & 1:
                            valid = False
                            break
                        new_row_diffs |= (1 << d)
                    
                    if not valid:
                        continue
                    
                    # Place x
                    row.append(x)
                    row_used_diffs |= new_row_diffs
                    
                    # Recurse
                    res, res_diffs = build_row_recursive(col_idx + 1, max_allowed)
                    if res is not None:
                        return res, res_diffs
                    
                    # Undo
                    row.pop()
                    row_used_diffs ^= new_row_diffs
                
                return None, None
            
            res, res_diffs = build_row_recursive(1, limit)
            if res is not None:
                return res, res_diffs
            
            # If we failed, increase the limit
            limit = int(limit * 1.2) + 10
            if limit > 2000: # Hard cap to prevent infinite loop
                return None, limit

    def greedy_full_solution(seed):
        """
        Builds all rows using a greedy approach.
        To minimize the max entry, we should build rows that "fit" well together.
        Strategy: Build rows one by one. For each new row, try to fit it into the
        remaining difference space.
        To keep the max small, we should prioritize using small differences early?
        Or spread them out?
        
        A common heuristic for difference sets is to start with a "dense" row (small diffs)
        and then subsequent rows must use larger diffs. This tends to increase the max quickly.
        
        Alternative: Randomized greedy where we shuffle the order of rows? No, rows are symmetric.
        
        Let's try: Build row 0. Then row 1. etc.
        To improve, we can use a "balanced" heuristic: when building a row, prefer
        differences that are not in the "hot zone" of the most recent rows?
        
        For simplicity and speed, we'll use the standard greedy:
        1. used_diffs = 0
        2. For each row:
           a. Estimate min possible max for this row given used_diffs.
           b. Build row with that limit.
        """
        rng = random.Random(seed)
        rows = []
        used_diffs = 0
        
        for r in range(n_rows):
            # Estimate limit
            # We need diffs_per_row new differences.
            # The smallest possible max value for a row with D distinct diffs not in U
            # is roughly the D-th number not in U.
            
            # Find the D-th smallest number not in U
            count = 0
            d = 1
            estimated_max = 0
            while count < diffs_per_row:
                if not ((used_diffs >> d) & 1):
                    count += 1
                    if count == diffs_per_row:
                        estimated_max = d
                        break
                d += 1
            
            # Add some slack
            limit = estimated_max + 10 + rng.randint(0, 5)
            
            row, row_diffs = greedy_build_row(used_diffs, limit, rng, k)
            
            if row is None:
                # If it failed with the tight limit, try a larger one
                # This shouldn't happen often if estimated_max is correct
                row, row_diffs = greedy_build_row(used_diffs, estimated_max + 50, rng, k)
                
            if row is None:
                return None, None, None # Failure
            
            rows.append(row)
            used_diffs |= row_diffs
            
            if get_time_left() < 0.5:
                return None, None, None # Timeout
        
        max_entry = max(max(row) for row in rows)
        return rows, used_diffs, max_entry

    def repair_row(row_idx, rows, used_diffs_all, seed):
        """
        Attempts to reconstruct row_idx to have a smaller max entry.
        It removes the differences of row_idx from the global used_diffs,
        then tries to build a new row that fits in the available space,
        aiming for a max entry smaller than the current one.
        """
        rng = random.Random(seed)
        
        # Calculate used_diffs without this row
        # We need the diff bitset for the current row
        current_row = rows[row_idx]
        current_row_diffs = 0
        for i in range(len(current_row)):
            for j in range(i + 1, len(current_row)):
                d = current_row[j] - current_row[i]
                current_row_diffs |= (1 << d)
        
        available_diffs = used_diffs_all ^ current_row_diffs
        
        current_max = current_row[-1]
        target_max = current_max - 1
        
        # We want to find a row with max <= target_max
        # If we find one with max < current_max, it's an improvement.
        
        # Try to build a row with max <= target_max
        # We can use the same greedy_build_row, but we must ensure it stays under target_max.
        # My greedy_build_row already respects the limit.
        
        # To make it more robust, we can try multiple times with slight randomization
        # in the candidate order? The current build_row_recursive is deterministic increasing.
        # Let's add a bit of randomness: if we fail, shuffle the preference?
        # For now, let's just call it.
        
        new_row, new_diffs = greedy_build_row(available_diffs, target_max, rng, k)
        
        if new_row is not None and new_row[-1] < current_max:
            # Success
            rows[row_idx] = new_row
            # Update global used_diffs
            new_used_diffs = used_diffs_all ^ current_row_diffs | new_diffs
            return True, new_used_diffs, new_row[-1]
        
        return False, used_diffs_all, current_max

    # Main Loop
    # 1. Generate initial solutions
    num_initial = 10
    for i in range(num_initial):
        if get_time_left() < 2.0:
            break
        rows, used_diffs, max_val = greedy_full_solution(i)
        if rows is not None:
            if max_val < best_max:
                best_max = max_val
                best_rows = rows
                best_used_diffs = used_diffs
                #print(f"Init {i}: max {max_val}") # Debug

    if best_rows is None:
        # Fallback: simple greedy without optimization
        rows, _, _ = greedy_full_solution(0)
        if rows is not None:
            return rows
        # If even that fails, return a dummy valid solution? 
        # The task requires a valid solution. 
        # A very simple fallback:
        # Row 0: 0, 1, 3, 6, ... (triangular)
        # This might not be valid for multiple rows.
        # But for n_rows=4, k=7, we need a valid set.
        # Let's hope greedy_full_solution works.
        # If it doesn't, we have a problem.
        # Let's try a larger limit in greedy_full_solution if it fails.
        for seed in range(10):
            rows, used_diffs, max_val = greedy_full_solution(seed)
            if rows is not None:
                return rows
        # Absolute fallback: return a list of lists that might be invalid?
        # No, must be valid.
        # Let's construct a trivial valid set if possible.
        # For small n, k, we can hardcode? No, n and k are passed.
        # Let's try greedy with a very large limit.
        rows, _, _ = greedy_full_solution(999)
        if rows:
            return rows
        return [[0]] * n_rows # Invalid, but prevents crash. Score 0.

    # 2. Local Search / Repair Phase
    # We will iterate, picking the row with the largest entry and trying to shrink it.
    # We also try to shuffle the rows to break symmetry? No, rows are indistinguishable.
    # But the *set* of used differences matters.
    
    # To avoid getting stuck in a local optimum where no single row can be shrunk,
    # we can try "pairwise swaps" or "re-randomizing" one row completely.
    
    # Strategy:
    # While time left:
    #   1. Find row with max entry.
    #   2. Try to repair it to be smaller.
    #   3. If successful, update best.
    #   4. If failed, try to repair a random row (to escape local minima).
    #   5. If too many failures, re-initialize with a new seed.
    
    fail_count = 0
    max_fails = 20
    
    while get_time_left() > 1.0:
        if fail_count >= max_fails:
            # Restart with a new initial solution
            seed = int(time.time() * 1000) % 10000
            rows, used_diffs, max_val = greedy_full_solution(seed)
            if rows is not None and max_val < best_max:
                best_max = max_val
                best_rows = rows
                best_used_diffs = used_diffs
                fail_count = 0
            else:
                # Keep the old best, but reset fail count to try again
                fail_count = 0
                continue
        
        # Pick a row to repair
        # 80% chance: pick the row with the largest entry
        # 20% chance: pick a random row
        if random.random() < 0.8:
            # Find index of max row
            max_val_in_rows = [max(row) for row in best_rows]
            target_idx = max_val_in_rows.index(max(max_val_in_rows))
        else:
            target_idx = random.randint(0, n_rows - 1)
            
        # Attempt repair
        success, new_used_diffs, new_max = repair_row(target_idx, best_rows, best_used_diffs, int(time.time()*10000)%10000)
        
        if success:
            best_used_diffs = new_used_diffs
            # Recalculate global max
            current_global_max = max(max(row) for row in best_rows)
            if current_global_max < best_max:
                best_max = current_global_max
                fail_count = 0 # Reset fail count on success
            else:
                # Improved one row, but global max didn't drop? 
                # This happens if another row was the max.
                fail_count = 0 # Still progress
        else:
            fail_count += 1
            
    return best_rows

# Helper to run
if __name__ == "__main__":
    # Test with the specific case from the prompt
    # n_rows=4, k=7
    res = construct(4, 7)
    print(f"Result: {res}")
    print(f"Max: {max(max(row) for row in res)}")
    
    # Verify validity
    used = set()
    valid = True
    for row in res:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                    print(f"Conflict: {d} in {row}")
                else:
                    used.add(d)
    print(f"Valid: {valid}")
