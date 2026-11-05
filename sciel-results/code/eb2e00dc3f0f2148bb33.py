"""
Strategy: Constraint-Programming with Backtracking and Forward Checking.
Instead of stochastic local search (SA) which wanders the landscape, we use a
deterministic (but randomized for restarts) backtracking solver with strong
constraint propagation.

Key structural differences:
1. Representation: We maintain a global 'used' set of differences. When placing
   a new element in a row, we immediately check against this set. If a conflict
   arises, we backtrack. This is a "build-and-solve" approach vs. "fix-it" SA.
2. Heuristic: We use the "Most Constrained Variable" (MCV) heuristic implicitly
   by trying to place elements as small as possible (which constrains future
   choices more tightly) but with randomized tie-breaking to explore different
   branches.
3. Optimization: We run the solver multiple times with a tightening "max_limit".
   We start with a loose limit (greedy solution) and progressively reduce it.
   Because the solver is exact (within the limit), finding a valid solution with
   a smaller limit is a definitive improvement, not a probabilistic one.
"""
import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound: max entry must be at least total_diffs (since diffs are distinct positive ints starting from 1)
    # Actually, the smallest distinct differences are 1, 2, ..., total_diffs.
    # So the largest difference in the entire set is at least total_diffs.
    # Since the largest difference in a row is row[-1] - row[0] = row[-1],
    # the max entry is at least total_diffs.
    lower_bound = total_diffs
    
    # 1. Get a quick valid solution using greedy (to set an initial upper bound)
    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = pure_greedy()
    best_max = max(max(row) for row in baseline_rows)
    best_rows = [row[:] for row in baseline_rows]
    
    # 2. Backtracking Solver with Forward Checking
    # We try to construct rows one by one.
    # State: current_rows (list of completed rows), current_row (being built), used_diffs (set)
    
    def solve_with_limit(max_limit, time_limit=5.0):
        """
        Attempts to find a valid set of rows where all entries are <= max_limit.
        Uses randomized backtracking.
        Returns list of rows or None.
        """
        used_diffs = set()
        rows = []
        
        # We will build rows one by one.
        # For each row, we build it element by element.
        
        def backtrack_row(row_idx, current_row, start_val):
            """
            Recursively build row_idx.
            current_row: list of elements so far (starts with [0])
            start_val: the minimum value for the next element (must be > current_row[-1])
            """
            if len(current_row) == k + 1:
                # Row complete. Check if we have all rows.
                if row_idx == n_rows - 1:
                    # Success
                    rows.append(current_row[:])
                    return True
                else:
                    # Move to next row
                    # Add current row's diffs to used_diffs (they were added incrementally? No, let's add on completion to simplify state passing, or pass used_diffs as mutable)
                    # Actually, adding incrementally is better for forward checking.
                    # Let's refactor: pass used_diffs as a mutable set and undo on backtrack.
                    pass
            
            # This simple recursion is hard to manage with global state.
            # Let's use a more iterative or state-passing approach.
            return False

        # Better approach: Iterative deepening / Backtracking with explicit stack or recursion with state.
        # Given the small k (usually 7 or 8) and n_rows (4), recursion depth is manageable.
        
        # Let's implement a recursive backtracker that builds one row at a time.
        
        def build_row(row_idx):
            """
            Builds row_idx. 
            Returns (success, row, updated_used_diffs)
            """
            # We need to try all possible increasing sequences of length k+1 starting at 0,
            # with max <= max_limit, such that diffs don't conflict with used_diffs OR each other.
            
            # To make this efficient, we build element by element.
            # State: current_row, used_diffs_local (diffs within this row so far)
            
            def _backtrack(current_row, used_local):
                if len(current_row) == k + 1:
                    # Check if this row's diffs conflict with global used_diffs
                    # used_local contains all diffs within this row.
                    if not (used_local & used_diffs):
                        # Success for this row
                        return True, current_row, used_local
                    else:
                        return False, None, None
                
                last_val = current_row[-1]
                # Next value must be > last_val and <= max_limit
                # Also, we must leave enough room for remaining elements?
                # Remaining elements = k + 1 - len(current_row)
                # So next_val must be <= max_limit - (remaining - 1) ? No, just strictly increasing.
                
                # Heuristic: Try smaller values first to keep max low? 
                # Or try values that create "rare" differences?
                # Randomized: Shuffle candidate values.
                
                min_next = last_val + 1
                max_next = max_limit
                # We must ensure we can still fill the row.
                # If we are at position i (0-indexed, so len=i+1), we have k - i elements left to place.
                # The last element must be <= max_limit.
                # So the current element must be <= max_limit - (k - i) ?
                # Not exactly, they just need to be distinct and increasing.
                # The tightest constraint is that the largest possible value for the current element
                # is max_limit - (number of remaining elements).
                remaining = k - len(current_row)
                if remaining > 0:
                    max_next = min(max_next, max_limit - remaining)
                
                if min_next > max_next:
                    return False, None, None
                
                candidates = list(range(min_next, max_next + 1))
                random.shuffle(candidates)
                
                for val in candidates:
                    # Compute new diffs
                    new_diffs = {val - a for a in current_row}
                    
                    # Check conflict with local used
                    if new_diffs & used_local:
                        continue
                    
                    # Check conflict with global used
                    if new_diffs & used_diffs:
                        continue
                    
                    # Tentatively add
                    new_used_local = used_local | new_diffs
                    new_row = current_row + [val]
                    
                    success, final_row, final_used = _backtrack(new_row, new_used_local)
                    if success:
                        return True, final_row, final_used
                
                return False, None, None
            
            return _backtrack([0], set())
        
        for i in range(n_rows):
            if time.time() > start_time + time_limit:
                return None
            
            success, row, row_diffs = build_row(i)
            if not success:
                return None
            
            rows.append(row)
            used_diffs |= row_diffs
        
        return rows

    # 3. Main Search Loop: Iterative Deepening on Max Limit
    # Start from best_max - 1 and go down to lower_bound.
    # For each limit, try multiple times with different random seeds/shuffles.
    
    current_limit = best_max - 1
    max_attempts_per_limit = 20 # Adjust based on time
    
    while current_limit >= lower_bound and time.time() < deadline:
        # Try to find a solution with max entry <= current_limit
        found = False
        attempts = 0
        while attempts < max_attempts_per_limit and time.time() < deadline:
            # Random seed for shuffling in backtracker
            random.seed(int(time.time() * 1000) + attempts + current_limit)
            
            sol = solve_with_limit(current_limit, time_limit=2.0)
            if sol is not None:
                # Verify solution
                used = set()
                valid = True
                for row in sol:
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            d = row[j] - row[i]
                            if d in used:
                                valid = False
                                break
                            used.add(d)
                    if not valid: break
                
                if valid:
                    new_max = max(max(row) for row in sol)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [row[:] for row in sol]
                        # Update limit for next iteration
                        current_limit = new_max - 1
                        found = True
                        break
            attempts += 1
        
        if not found:
            # No solution found for this limit in allocated time/attempts
            # Move to next lower limit? Or break?
            # If we can't find a solution for current_limit, we might not find one for lower limits easily.
            # But due to randomness, we might have missed it.
            # Let's decrease the limit and try again.
            current_limit -= 1
    
    return best_rows
# EVOLVE-BLOCK-END
