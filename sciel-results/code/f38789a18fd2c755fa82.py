"""
Strategy: Randomized Backtracking with Forward Checking.
1. Instead of greedy construction followed by local repair, we construct rows 
   one by one using backtracking.
2. We maintain a global set of used differences.
3. When choosing a value for a position in a row, we check if it creates 
   conflicts with the global pool.
4. We use a heuristic to order rows or values to find compact solutions faster.
5. We restart with different random seeds to explore the solution space, 
   keeping the best (smallest max entry) found.
"""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Trivial cases
    if n_rows == 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    best_rows = None
    best_score = float('inf')
    
    # We will try multiple random restarts
    # The number of restarts is not fixed, but determined by time budget
    # Each restart attempts to find a valid solution with a specific max limit M
    # We start with a generous M and try to reduce it, or just let the solver
    # find the minimal M naturally by trying small values first.
    
    # A simple greedy construction gives a baseline
    baseline_rows = _greedy_construct(n_rows, k)
    baseline_score = _max_entry(baseline_rows)
    best_rows = baseline_rows
    best_score = baseline_score
    
    # Now try to improve with backtracking
    # We will try to find a solution with max entry < current best_score
    # We do this by setting a limit L and seeing if a solution exists with max <= L.
    # We start L just below best_score and go down? Or just run the solver 
    # which prefers smaller values.
    
    # The solver will try to build rows. 
    # To make it effective, we should probably constrain the max value 
    # to something reasonable, or let it grow but prefer smaller.
    # Since the greedy baseline is already valid, we know a solution exists 
    # at baseline_score.
    
    # Let's try to find a solution with max < best_score.
    # We can do binary search on M, or just iterative deepening.
    # Given the time limit, iterative deepening from a lower bound up to best_score-1 
    # might be too slow if the gap is large.
    # However, the gap is usually small for these problems (e.g., 112 vs 105).
    
    # Let's try a randomized backtracking that tries to keep the max entry low.
    # We'll run the solver multiple times with randomization in the value ordering.
    
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        # Try to find a solution with max < best_score
        # We set a limit M = best_score - 1
        M = best_score - 1
        if M < k * k: # Lower bound heuristic
            break
            
        # Run backtracking with limit M
        # We want to find ANY valid set of rows with max <= M
        # If found, update best_score to the actual max of that set (which might be < M)
        # Actually, if we find a valid set, its max is <= M. 
        # We want to minimize the max. 
        # The backtracking solver, if it prefers smaller values, will likely 
        # find a solution with max close to the minimum possible for that M.
        # But we want the global minimum max.
        
        # Let's modify the solver to return the actual max used.
        # If it succeeds with limit M, we get a valid set.
        # We can then set best_score to the max of that set.
        # Then try M = new_best_score - 1.
        
        result = _backtrack_solve(n_rows, k, M, deadline)
        if result is not None:
            actual_max = _max_entry(result)
            if actual_max < best_score:
                best_score = actual_max
                best_rows = result
            else:
                # If it didn't improve, maybe try a tighter limit next time?
                # Or just break if we can't go lower?
                # If actual_max == M, we might try M-1 next.
                # If actual_max < M, we are already at a lower bound for this search.
                pass
        
        # If we haven't improved in several attempts, maybe the limit M is too low 
        # for the current randomization, or we are stuck.
        # But since we reduce M only on success, and start from best_score-1, 
        # we are effectively doing a descent.
        
        # To avoid infinite loops if no solution exists for M but exists for M+1,
        # we must be careful. But we know a solution exists for best_score (the baseline).
        # So if M = best_score - 1 fails, we should stop trying lower M?
        # No, because randomization might find a different configuration.
        # But if M is the absolute minimum, it will never succeed.
        # How to detect this? We can't easily.
        # So we just run until time runs out, hoping to find a better one.
        # If we find one, we lower M.
        
        # If we fail to find any solution for M after some attempts, 
        # we might want to increase M? No, we want to lower the max.
        # If M is too low, no solution exists. 
        # Since we start from a known valid max (best_score), and decrement, 
        # we will eventually hit the true minimum.
        # If we reach the true minimum, we will keep failing.
        # So we should stop if we fail too many times in a row for the same M?
        # Or just let it run? 
        # Let's add a stagnation check.
        
        if attempts % 10 == 0:
             # Check if we are still making progress
             pass

    return best_rows

def _greedy_construct(n_rows, k):
    used = set()
    rows = []
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

def _max_entry(rows):
    return max(max(row) for row in rows)

def _backtrack_solve(n_rows, k, limit, deadline):
    """
    Attempts to construct n_rows rows of k+1 integers, starting at 0, increasing,
    with max entry <= limit, such that all differences are distinct.
    Returns the list of rows if successful, None otherwise.
    Uses randomization to escape local minima.
    """
    used_diffs = set()
    rows = []
    
    # We construct rows one by one.
    # To make it more likely to find a compact solution, we can shuffle the order 
    # of rows? No, rows are indistinguishable in the final set, but the construction 
    # order matters for the "used_diffs" pool.
    # Actually, the problem is symmetric with respect to rows.
    # We can just construct row 0, then row 1, etc.
    
    # Heuristic: Construct rows with the largest "demand" first? 
    # All rows have the same demand (k+1 elements).
    # Maybe randomizing the order of values within the row helps?
    
    # Let's try a recursive backtracking for each row.
    # Since rows are independent except for the shared diff pool, 
    # we can solve for one row at a time.
    
    def solve_row(row_idx, current_row, pos):
        if time.time() > deadline - 1:
            return False
            
        if pos == k:
            # Row is complete
            rows.append(current_row[:])
            return True
            
        # Determine range for current value
        min_val = current_row[-1] + 1
        max_val = min(limit, min_val + (k - pos) * 10) # Heuristic upper bound to keep it tight
        
        # Randomize the order of candidate values to avoid getting stuck
        # in a specific pattern. We prefer smaller values, but with some noise.
        candidates = list(range(min_val, max_val + 1))
        
        # Sort by a heuristic: prefer values that create "rare" differences?
        # Or just shuffle slightly?
        # Let's try: 80% chance to pick from the bottom half, 20% from anywhere.
        # Or simpler: Shuffle the list, but bias towards smaller.
        # We'll just shuffle and iterate.
        random.shuffle(candidates)
        
        # To maintain some determinism in the "search for minimum", 
        # we might want to try smaller values first more often.
        # Let's sort by value, but with a random tie-breaker? 
        # No, values are unique.
        # Let's just iterate through candidates in a randomized order 
        # but weighted towards smaller values.
        
        # Simple approach: Shuffle, then sort by (value + random_noise)
        # Or just shuffle.
        
        for val in candidates:
            if val > limit:
                continue
                
            # Check internal conflicts
            new_diffs = set()
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
                
            if not valid:
                continue
                
            # Check global conflicts
            if new_diffs & used_diffs:
                continue
                
            # Accept
            current_row.append(val)
            used_diffs.update(new_diffs)
            
            if solve_row(row_idx, current_row, pos + 1):
                return True
            
            # Backtrack
            current_row.pop()
            used_diffs.difference_update(new_diffs)
            
        return False

    # Main loop for rows
    for i in range(n_rows):
        current_row = [0]
        if not solve_row(i, current_row, 1):
            return None
            
    return rows

# Helper for validity check of the whole set (used implicitly in logic)
def _is_valid(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

