import time
import random
import sys

def construct(n_rows, k):
    # Calculate the theoretical lower bound for the maximum entry M
    # Total distinct differences needed: n_rows * C(k+1, 2) = n_rows * k * (k+1) / 2
    # Maximum distinct differences available in [1, M] is M * (M+1) / 2
    # So M * (M+1) / 2 >= n_rows * k * (k+1) / 2
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Solve M(M+1) >= 2 * total_diffs_needed
    # M approx sqrt(2 * total_diffs_needed)
    M_lower = int((2 * total_diffs_needed) ** 0.5)
    while M_lower * (M_lower + 1) // 2 < total_diffs_needed:
        M_lower += 1
    
    # We will search for a solution with max entry M.
    # Start from the lower bound and increase M if no solution is found.
    # The score is the max entry, so we want the smallest M possible.
    
    # Time management
    # The evaluator kills after 150s. We should aim to finish well before.
    # Let's use a main deadline of 140s to be safe.
    main_deadline = time.time() + 140.0
    
    best_rows = None
    best_M = float('inf')
    
    # Bitmask size: We need to represent differences up to M.
    # If M is around 100-200, we can use Python integers (arbitrary precision) as bitmasks.
    # diff d corresponds to bit (d-1).
    
    def try_solve(M, time_limit):
        """
        Attempts to find a valid set of rows with max entry <= M.
        Returns a list of rows if found, None otherwise.
        """
        # Pre-compute bitmask for differences for all possible values x in 1..M
        # mask[x] has bits set for differences x-a for a in 0..x-1? 
        # No, the differences depend on the current row content.
        # However, we can pre-compute the "potential" difference masks for a single element
        # if we knew the previous elements. Since previous elements vary, we compute on the fly.
        
        # Optimization: Pre-compute a list of all possible difference masks for a value x
        # given that the row starts with 0? No, the row grows.
        
        # Let's use a recursive backtracking function.
        
        # To speed up the forward check, we might need to be efficient.
        
        used_diffs_mask = 0
        rows = []
        
        # We need to build n_rows rows.
        # Each row has k+1 elements: 0, x1, ..., xk.
        
        def dfs_row(row_idx, pos, current_row, current_diffs_mask):
            """
            Recursively builds the row at index row_idx.
            pos: current number of elements in current_row (including 0).
            current_row: list of elements in the current row.
            current_diffs_mask: bitmask of differences used within this row so far.
            """
            if time.time() > time_limit:
                return False
            
            if pos == k + 1:
                # Row complete
                rows.append(current_row[:])
                # Update global used_diffs_mask? 
                # We pass it by reference (nonlocal) or return it.
                # For efficiency, we'll use nonlocal variables for the state.
                nonlocal used_diffs_mask
                used_diffs_mask |= current_diffs_mask
                
                if row_idx + 1 == n_rows:
                    return True
                # Start next row
                return dfs_row(row_idx + 1, 1, [0], 0)
            
            # Generate candidates for the next element
            last = current_row[-1]
            lo = last + 1
            hi = M
            
            # Forward checking: We need to ensure we can fill the remaining (k+1 - pos) positions.
            # Simple check: hi - lo + 1 >= k + 1 - pos
            
            candidates = []
            
            # To make the search deterministic-ish but varied, we can collect all valid candidates
            # and then shuffle or sort them.
            
            for x in range(lo, hi + 1):
                # Calculate differences with all previous elements in current_row
                # d = x - a for a in current_row
                # Check if any d is in used_diffs_mask or current_diffs_mask
                # If so, skip.
                
                # Optimization: 
                # We can check (diff_mask & (used_diffs_mask | current_diffs_mask)) == 0
                
                diff_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    bit = 1 << (d - 1)
                    if diff_mask & bit:
                        # Duplicate difference within the row?
                        # This shouldn't happen if x is strictly increasing and we check against current_diffs_mask
                        # But wait, if a1 < a2, x-a1 > x-a2. So bits are distinct.
                        pass
                    diff_mask |= bit
                    
                    if (diff_mask & (used_diffs_mask | current_diffs_mask)) != 0:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Forward Check: Can we place at least one more element after x?
                # If pos + 1 < k + 1 (i.e., there are more elements to add):
                if pos + 1 < k + 1:
                    # Check if there exists at least one y > x such that y is valid
                    # Given current_row + [x]
                    # This check can be expensive. 
                    # Simple heuristic: if M - x < (k + 1 - (pos + 1)), it's impossible.
                    remaining_slots = k + 1 - (pos + 1)
                    if M - x < remaining_slots:
                        continue
                    
                    # Optional: Deeper forward check (expensive)
                    # For now, let's rely on the simple space check and backtracking.
                    # The bitmask check is fast enough.
                
                candidates.append(x)
            
            if not candidates:
                return False
            
            # Heuristic ordering:
            # We want to pick values that leave the most room for future values.
            # Smaller x leaves more room (M-x is larger).
            # However, sometimes smaller x uses "popular" small differences.
            # Let's sort by x ascending, but add some randomness.
            
            # To avoid getting stuck in the same local minimum on every restart,
            # we can shuffle the candidates.
            random.shuffle(candidates)
            
            # Or sort by a score.
            # Score: prefer smaller x?
            # Let's try sorting by x, but with a random tie-breaker or partial shuffle.
            # Actually, just shuffling is good for diversification.
            
            for x in candidates:
                # Recurse
                new_diff_mask = 0
                for a in current_row:
                    d = x - a
                    new_diff_mask |= (1 << (d - 1))
                
                # Update state
                current_row.append(x)
                new_current_diffs_mask = current_diffs_mask | new_diff_mask
                
                if dfs_row(row_idx, pos + 1, current_row, new_current_diffs_mask):
                    return True
                
                # Backtrack
                current_row.pop()
            
            return False
        
        # Start DFS
        # Reset state
        used_diffs_mask = 0
        rows = []
        
        # The nonlocal declaration inside nested functions in Python 3 works, 
        # but passing state via arguments or using a class/struct is cleaner.
        # However, for speed, let's stick to the closure if possible, 
        # or just pass the mask as an argument and return the new mask?
        # The issue is `used_diffs_mask` is global across rows.
        
        # Let's refactor to pass `used_diffs_mask` as an argument to `dfs_row` 
        # and return the updated mask if successful? 
        # No, `dfs_row` modifies `rows` and `used_diffs_mask`.
        
        # Let's use a simple object or just rely on the fact that we are in a single-threaded context
        # and use a mutable container for the state.
        
        state = {
            'used_diffs_mask': 0,
            'rows': []
        }
        
        def dfs_row_state(row_idx, pos, current_row, current_diffs_mask):
            if time.time() > time_limit:
                return False
            
            if pos == k + 1:
                state['rows'].append(current_row[:])
                state['used_diffs_mask'] |= current_diffs_mask
                
                if row_idx + 1 == n_rows:
                    return True
                return dfs_row_state(row_idx + 1, 1, [0], 0)
            
            last = current_row[-1]
            lo = last + 1
            hi = M
            
            candidates = []
            
            for x in range(lo, hi + 1):
                diff_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    bit = 1 << (d - 1)
                    diff_mask |= bit
                    if (diff_mask & (state['used_diffs_mask'] | current_diffs_mask)) != 0:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Simple forward check: space
                remaining_slots = k + 1 - (pos + 1)
                if pos + 1 < k + 1 and M - x < remaining_slots:
                    continue
                
                candidates.append(x)
            
            if not candidates:
                return False
            
            random.shuffle(candidates)
            
            for x in candidates:
                new_diff_mask = 0
                for a in current_row:
                    d = x - a
                    new_diff_mask |= (1 << (d - 1))
                
                current_row.append(x)
                new_current_diffs_mask = current_diffs_mask | new_diff_mask
                
                if dfs_row_state(row_idx, pos + 1, current_row, new_current_diffs_mask):
                    return True
                
                current_row.pop()
            
            return False
        
        if dfs_row_state(0, 1, [0], 0):
            return state['rows']
        return None

    # Main Loop: Iterate M starting from lower bound
    M = M_lower
    restarts_per_M = 0
    max_restarts_per_M = 50 # Limit restarts for a single M to avoid wasting time
    
    while time.time() < main_deadline:
        # Allocate time for this M
        # If we have little time left, be strict.
        time_left = main_deadline - time.time()
        
        # Give each restart a small time slice, e.g., 2 seconds?
        # Or just run until we find a solution or exhaust restarts.
        # Let's use a dynamic time limit per attempt.
        attempt_time_limit = min(2.0, time_left / 10.0) # At least 10 attempts possible in remaining time
        
        found = False
        for r in range(max_restarts_per_M):
            if time.time() > main_deadline:
                break
            
            random.seed() # New random state for each restart
            
            rows = try_solve(M, time.time() + attempt_time_limit)
            if rows is not None:
                # Valid solution found
                best_rows = rows
                best_M = M
                found = True
                break
        
        if found:
            # We found a solution for M. Can we find a smaller one?
            # We started from the lower bound, so M is the smallest possible we haven't tried yet?
            # Actually, we increment M. So if we found a solution for M, it's the best for this M.
            # But maybe a solution exists for M-1 that we missed?
            # The problem is NP-hard. We don't know if M-1 is feasible.
            # However, the score is the max entry.
            # If we found a solution for M, we return it.
            # To potentially improve, we would need to prove M-1 is infeasible or search it better.
            # Given the time limit, returning the first valid M found (starting from lower bound) is a good strategy.
            # If we find one for M_lower, that's optimal (or close to it).
            # If we find one for M_lower+1, that's the best we have.
            
            # Wait, the loop increments M.
            # If we find a solution at M=112, we return it.
            # If we don't find it at M=110, we try M=111.
            # If we find it at M=111, we return it.
            # This is correct.
            
            return best_rows
        
        # If no solution found for this M, increase M
        M += 1
        # If M gets too large, we might be in trouble, but the greedy fallback handles it.
        # Also, we should stop if M is unreasonably large?
        # The greedy fallback will produce a valid result, possibly with a larger M.
        # We should compare the result of the search vs the greedy fallback.
        
        # Actually, let's just keep going until time is up or we find a solution.
        # If time runs out, we fall back to greedy.

    # Fallback: Greedy construction
    # This ensures we always return a valid result.
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
    
    # Compare greedy result with best found (if any)
    if best_rows is not None:
        greedy_max = max(max(r) for r in rows)
        search_max = max(max(r) for r in best_rows)
        if greedy_max < search_max:
            return rows
        else:
            return best_rows
    else:
        return rows

if __name__ == "__main__":
    # Test with the specific task parameters
    # n_rows = 7, k = 5
    # This is for local testing, the evaluator calls construct(n_rows, k)
    rows = construct(7, 5)
    print("Max entry:", max(max(r) for r in rows))
    print("Rows:", rows)
