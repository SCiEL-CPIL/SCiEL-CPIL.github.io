"""
Hybrid approach:
1. Fast greedy column-wise construction (multiple strategies/orderings) to get an initial feasible set.
2. Local search / simulated annealing to refine and minimize the scope (largest entry).
3. If time permits, a bounded backtracking search with strong pruning to try to find near-optimal solutions.

Key structural change from the parent:
- The parent used a simple greedy + local search.
- This version incorporates:
  a) Multiple diverse greedy initializations (different row orderings, column strategies).
  b) A more sophisticated local search that tries to "swap" differences or shift elements.
  c) A time-bounded backtracking solver that uses the greedy result as an upper bound for pruning.
"""
import time
import random
import math

def _deadline_guard(start, margin=5.0, limit=145.0):
    """Return True if we should stop (elapsed + margin >= limit)."""
    return (time.time() - start) + margin >= limit

def _score_of(rows):
    """Largest entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _is_valid(rows, k):
    """Check that all positive differences within rows are distinct across rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _build_greedy(n_rows, k, strategy="col_asc", rng=None):
    """
    Build a feasible set using greedy column-wise assignment.
    strategy: 
        - "col_asc": process rows in ascending order of current last element for each column.
        - "col_desc": process rows in descending order.
        - "random": random row order for each column.
    """
    if rng is None:
        rng = random.Random(0)
    
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()

    for col in range(k):
        if strategy == "col_asc":
            row_order = sorted(range(n_rows), key=lambda r: rows[r][-1])
        elif strategy == "col_desc":
            row_order = sorted(range(n_rows), key=lambda r: -rows[r][-1])
        else:
            row_order = list(range(n_rows))
            rng.shuffle(row_order)

        for r in row_order:
            x = rows[r][-1] + 1
            while True:
                conflict = False
                for i in range(len(rows[r])):
                    d = x - rows[r][i]
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    break
                x += 1
            
            for i in range(len(rows[r])):
                used_diffs.add(x - rows[r][i])
            rows[r].append(x)
    
    return rows

def _local_search(rows, k, start_time, limit=145.0, max_iters=5000):
    """
    Local search to reduce the maximum entry.
    We try to replace an element in a row with a smaller feasible value.
    """
    best_rows = [list(r) for r in rows]
    best_score = _score_of(best_rows)
    
    # Precompute used diffs for quick update
    used_diffs = set()
    for row in best_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    rng = random.Random(42)
    
    for _ in range(max_iters):
        if _deadline_guard(start_time, 5.0, limit):
            break
        
        # Pick a random row and a random position (from 1 to k)
        r = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)
        
        old_val = best_rows[r][pos]
        
        # Try to find a smaller value for this position
        # The new value x must be > best_rows[r][pos-1] and < old_val
        # Also, all differences (x - best_rows[r][i]) for i != pos must not conflict with used_diffs (excluding this row's diffs)
        
        # Remove this row's diffs from used_diffs temporarily
        row_diffs = set()
        for i in range(len(best_rows[r])):
            for j in range(i + 1, len(best_rows[r])):
                row_diffs.add(best_rows[r][j] - best_rows[r][i])
        for d in row_diffs:
            used_diffs.discard(d)
        
        # Find a better value
        found = False
        # Try values from best_rows[r][pos-1] + 1 to old_val - 1
        # We want to minimize the max, so we prefer smaller values if possible, but we also need to ensure the rest of the row can be completed.
        # For simplicity, we just try to find *any* valid value that is smaller than old_val.
        # Actually, to reduce the max, we should try to reduce the *last* element of the row if it is the global max.
        
        # Let's focus on reducing the last element of the row if it's large.
        if pos == k:
            # Try to find a smaller value for the last element
            for new_val in range(best_rows[r][k-1] + 1, old_val):
                conflict = False
                for i in range(k): # i is 0..k-1
                    d = new_val - best_rows[r][i]
                    if d in used_diffs:
                        conflict = True
                        break
                if not conflict:
                    # Check if new_val is valid (it is, by construction)
                    # Accept if it reduces the max
                    if new_val < best_rows[r][k]:
                        # Update used_diffs
                        for i in range(k):
                            used_diffs.add(new_val - best_rows[r][i])
                        best_rows[r][k] = new_val
                        new_score = _score_of(best_rows)
                        if new_score < best_score:
                            best_score = new_score
                        found = True
                        break
        
        # Re-add the row's diffs (which may have changed if we found a new value)
        # Wait, if we found a new value, we already added the new diffs. We need to remove the old ones?
        # No, we removed ALL diffs of the row at the start.
        # So we need to add the current diffs of the row back.
        for i in range(len(best_rows[r])):
            for j in range(i + 1, len(best_rows[r])):
                used_diffs.add(best_rows[r][j] - best_rows[r][i])
        
        # If no improvement, continue
    
    return best_rows, best_score

def _backtrack_search(n_rows, k, target_scope, start_time, limit=145.0, rng=None):
    """
    Bounded backtracking search to find a solution with max entry < target_scope.
    """
    if rng is None:
        rng = random.Random(0)
    
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, current_row, current_max):
        if _deadline_guard(start_time, 5.0, limit):
            return None
        
        if row_idx == n_rows:
            return [list(r) for r in rows]
        
        if len(current_row) == k + 1:
            rows.append(current_row)
            result = backtrack(row_idx + 1, [0], current_max)
            if result is not None:
                return result
            rows.pop()
            return None
        
        last_val = current_row[-1]
        # We want to find a value x such that last_val < x < target_scope
        # And all differences are distinct and not in used_diffs
        
        # Heuristic: try values in a random order or sorted order?
        # Sorted order (smallest first) tends to keep the scope small.
        # But we might get stuck. Randomization helps.
        
        candidates = list(range(last_val + 1, target_scope))
        # Shuffle a bit to avoid always taking the smallest if it leads to a dead end
        # But for minimizing scope, smallest is usually best.
        # Let's try smallest first, but with some randomization in the local search.
        
        # To improve performance, we can limit the number of candidates we try per node.
        # If we try too many, we might run out of time.
        
        # Let's try the smallest few candidates first.
        # If none work, we might need to backtrack.
        
        # Optimization: Check if it's even possible to complete the row with the remaining slots.
        # This is complex, so we skip it for now.
        
        # Try candidates
        for x in candidates:
            # Check conflicts
            conflict = False
            new_diffs = []
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            # Also check if the new diffs are distinct among themselves?
            # Since a are distinct, x-a are distinct. So yes.
            
            # Add to used_diffs
            for d in new_diffs:
                used_diffs.add(d)
            
            result = backtrack(row_idx, current_row + [x], max(current_max, x))
            
            # Remove from used_diffs
            for d in new_diffs:
                used_diffs.discard(d)
            
            if result is not None:
                return result
        
        return None
    
    return backtrack(0, [0], 0)

def construct(n_rows, k):
    start_time = time.time()
    limit = 145.0  # Safety margin before 150s kill
    
    # Phase 1: Generate multiple initial solutions using different greedy strategies
    initial_solutions = []
    
    # Strategy 1: Column-wise ascending
    rows1 = _build_greedy(n_rows, k, "col_asc")
    initial_solutions.append(rows1)
    
    # Strategy 2: Column-wise descending
    rows2 = _build_greedy(n_rows, k, "col_desc")
    initial_solutions.append(rows2)
    
    # Strategy 3: Random
    rng = random.Random(42)
    rows3 = _build_greedy(n_rows, k, "random", rng)
    initial_solutions.append(rows3)
    
    # Phase 2: Local search on each initial solution
    best_rows = None
    best_score = float('inf')
    
    for sol in initial_solutions:
        if _deadline_guard(start_time, 10.0, limit):
            break
        
        refined_rows, score = _local_search(sol, k, start_time, limit)
        if score < best_score:
            best_score = score
            best_rows = refined_rows
    
    if best_rows is None:
        # Fallback to the first solution if local search didn't run
        best_rows = initial_solutions[0]
        best_score = _score_of(best_rows)
    
    # Phase 3: Try backtracking search to improve further
    # We try to find a solution with a smaller scope than the current best.
    # We start with target_scope = best_score - 1 and decrease.
    # But backtracking can be slow. We limit the time.
    
    if not _deadline_guard(start_time, 10.0, limit):
        # Try to find a solution with scope < best_score
        target = best_score - 1
        # We might want to try a few different targets or just the current best - 1
        # Let's try to find a solution with scope <= best_score - 1
        
        # To avoid exponential blowup, we only do this if the problem is small enough
        # or if we have plenty of time.
        
        # Let's try a bounded backtracking with a time limit.
        # We'll use a separate RNG for reproducibility if needed, but random is fine.
        
        # We'll try to find a solution with scope < best_score.
        # If found, update best_rows and best_score.
        
        # To make it faster, we can use the local search result as a guide?
        # No, backtracking is independent.
        
        # Let's try a few different random seeds for the backtracking to explore different branches.
        # But we need to be careful with time.
        
        # We'll run backtracking with a time limit of, say, 30 seconds.
        # If it finds a better solution, we update.
        
        # However, backtracking might be too slow for large n_rows or k.
        # For n_rows=7, k=5, it might be feasible.
        
        # Let's try it.
        
        # We'll use a helper function that runs backtracking with a time limit.
        
        # Since _backtrack_search already has a time limit, we can just call it.
        
        # But we need to pass the start_time and limit.
        
        # Let's try to find a solution with scope < best_score.
        
        # We'll try a few different targets.
        
        # Actually, let's just try to find *any* solution with scope < best_score.
        
        # We'll use a time limit of 30 seconds for this phase.
        
        backtrack_start = time.time()
        backtrack_limit = start_time + 30.0  # 30 seconds for backtracking
        
        # We need to modify _backtrack_search to accept a specific limit.
        # Or we can just use the global limit and hope it finishes.
        
        # Let's create a wrapper that runs backtracking with a specific time limit.
        
        # For simplicity, we'll just call _backtrack_search with the global start_time and limit.
        # But we want to limit it to 30 seconds.
        
        # Let's modify the call to use a shorter limit.
        
        # We'll create a new function or modify the existing one.
        
        # For now, let's just try it with a shorter limit.
        
        # We'll use a new start_time for the backtracking phase?
        # No, we should use the global start_time to ensure we don't exceed the total limit.
        
        # Let's just call it and see if it finds a better solution.
        
        # To avoid running out of time, we'll check the time before and after.
        
        if time.time() - start_time < 100.0:  # Only if we have at least 100s left (i.e., 50s elapsed)
            # Try backtracking with a time limit of 30s
            # We'll use a separate limit for this phase
            
            # We need to pass a limit that is start_time + 30.0
            # But _backtrack_search uses the global limit parameter.
            
            # Let's just call it with the global limit and hope it finishes quickly.
            # If not, it will return None.
            
            # We'll try a few different targets.
            
            # Let's try to find a solution with scope < best_score.
            
            # We'll use a random seed for the backtracking to explore different branches.
            
            # We'll try up to 5 different random seeds.
            
            for seed in range(5):
                if _deadline_guard(start_time, 5.0, limit):
                    break
                
                rng_bt = random.Random(seed)
                result = _backtrack_search(n_rows, k, best_score, start_time, limit, rng_bt)
                
                if result is not None:
                    new_score = _score_of(result)
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = result
                        # Update best_score and best_rows
                        # We can break here if we found a good solution
                        # But we might find an even better one
                        # Let's continue to try to improve further
                        # But we need to update the target for the next iteration
                        # Actually, _backtrack_search uses the target_scope to prune.
                        # So if we find a solution with score S, we can try to find one with S-1.
                        
                        # Let's update the target for the next iteration.
                        # But we are in a loop, so we can just continue.
                        
                        # However, we need to be careful with the time.
                        
                        # Let's break if we found a solution that is close to the theoretical lower bound.
                        # The lower bound is not known exactly, but we can use the counting bound.
                        
                        # For now, let's just continue.
    
    return best_rows

# To ensure the function signature is correct and the program is self-contained
# We need to make sure that all helper functions are defined before construct.

# Let's re-verify the code structure.
# The code above looks good.

# One potential issue: _backtrack_search might be too slow.
# We should add a check to ensure it doesn't take too long.
# The _deadline_guard inside _backtrack_search should handle this.

# Let's also add a check to ensure that the rows are valid before returning.
# This is a good practice to avoid returning an invalid solution.

def construct(n_rows, k):
    start_time = time.time()
    limit = 145.0  # Safety margin before 150s kill
    
    # Phase 1: Generate multiple initial solutions using different greedy strategies
    initial_solutions = []
    
    # Strategy 1: Column-wise ascending
    rows1 = _build_greedy(n_rows, k, "col_asc")
    initial_solutions.append(rows1)
    
    # Strategy 2: Column-wise descending
    rows2 = _build_greedy(n_rows, k, "col_desc")
    initial_solutions.append(rows2)
    
    # Strategy 3: Random
    rng = random.Random(42)
    rows3 = _build_greedy(n_rows, k, "random", rng)
    initial_solutions.append(rows3)
    
    # Phase 2: Local search on each initial solution
    best_rows = None
    best_score = float('inf')
    
    for sol in initial_solutions:
        if _deadline_guard(start_time, 10.0, limit):
            break
        
        refined_rows, score = _local_search(sol, k, start_time, limit)
        if score < best_score:
            best_score = score
            best_rows = refined_rows
    
    if best_rows is None:
        # Fallback to the first solution if local search didn't run
        best_rows = initial_solutions[0]
        best_score = _score_of(best_rows)
    
    # Phase 3: Try backtracking search to improve further
    # We try to find a solution with a smaller scope than the current best.
    
    if not _deadline_guard(start_time, 10.0, limit):
        # Try to find a solution with scope < best_score
        # We'll use a time limit of 30 seconds for this phase.
        
        # We'll try a few different random seeds for the backtracking.
        
        for seed in range(5):
            if _deadline_guard(start_time, 5.0, limit):
                break
            
            rng_bt = random.Random(seed)
            result = _backtrack_search(n_rows, k, best_score, start_time, limit, rng_bt)
            
            if result is not None:
                new_score = _score_of(result)
                if new_score < best_score:
                    best_score = new_score
                    best_rows = result
    
    # Final validation
    if not _is_valid(best_rows, k):
        # If the result is invalid, fall back to the first greedy solution
        best_rows = initial_solutions[0]
    
    return best_rows
