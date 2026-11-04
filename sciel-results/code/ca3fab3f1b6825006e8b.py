"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid_rows(rows):
    """Check that all rows are strictly increasing, start at 0, and have globally distinct differences."""
    all_diffs = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        row_d = _row_diffs(row)
        if row_d & all_diffs:
            return False
        all_diffs |= row_d
    return True


def _score(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_row(used, k, start=0, max_val=200):
    """Greedily build a row of k+1 elements starting at 0, using smallest feasible values."""
    row = [0]
    x = start
    while len(row) < k + 1:
        x += 1
        if x > max_val:
            return None
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row


def _local_search_row(row, used, k, max_val=300, iterations=5000):
    """Try to improve a single row by local search, given the used diffs from other rows."""
    if row is None:
        return None
    best = list(row)
    best_score = max(best)
    n = k + 1
    # We'll do a simple hill-climbing: pick a random position (not 0), try to change it
    # to a nearby value that keeps the row valid and reduces the max.
    rng = random.Random(42)
    for _ in range(iterations):
        cur = list(best)
        # Pick a random position to modify (positions 1..n-1)
        pos = rng.randint(1, n - 1)
        old_val = cur[pos]
        # Try values in a window around old_val, biased towards smaller
        window = rng.randint(1, 5)
        for delta in range(-window, window + 1):
            new_val = old_val + delta
            if new_val <= 0:
                continue
            # Check strict increase
            if pos > 0 and new_val <= cur[pos - 1]:
                continue
            if pos < n - 1 and new_val >= cur[pos + 1]:
                continue
            cur[pos] = new_val
            # Check differences
            cur_diffs = _row_diffs(cur)
            if cur_diffs & used:
                continue
            # Valid! Accept if it reduces the max or is equal
            if max(cur) < best_score:
                best = cur
                best_score = max(cur)
                break
    return best


def _solve(n_rows, k, deadline=140.0):
    """Main solver: build rows with backtracking + local search."""
    start_time = time.time()
    best_rows = None
    best_score = float('inf')
    
    # First, get a greedy baseline
    used = set()
    rows = []
    for i in range(n_rows):
        row = _greedy_row(used, k)
        if row is None:
            break
        rows.append(row)
        used |= _row_diffs(row)
    
    if len(rows) == n_rows:
        best_rows = [list(r) for r in rows]
        best_score = _score(best_rows)
    
    # Now try to improve using backtracking with limited depth
    # We'll try to rebuild rows one at a time, using backtracking for the last few rows
    
    # Strategy: fix first n_rows - 2 rows greedily, then do backtracking for last 2
    # Also try different orderings and restarts
    
    rng = random.Random(123)
    
    # Try multiple random restarts with perturbation
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Decide how many rows to fix greedily
        n_fixed = rng.randint(max(0, n_rows - 4), n_rows - 2)
        
        used = set()
        rows = []
        
        # Build n_fixed rows greedily with some randomization
        for i in range(n_fixed):
            # Add some randomness to the greedy choice
            row = [0]
            x = 0
            success = False
            while len(row) < k + 1:
                x += 1
                if x > 200:
                    break
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    # With some probability, skip this value to explore
                    if rng.random() < 0.1 and x < 150:
                        continue
                    row.append(x)
                    used |= new_diffs
                    success = True
            if not success:
                break
            rows.append(row)
            used |= _row_diffs(row)
        
        if len(rows) < n_fixed:
            continue
        
        # Now backtrack for remaining rows
        remaining = n_rows - len(rows)
        
        def backtrack(idx, used_set, cur_rows, max_allowed):
            nonlocal best_rows, best_score
            if time.time() > deadline - 3.0:
                return False
            if idx == remaining:
                # Check if this is better
                all_rows = cur_rows
                sc = _score(all_rows)
                if sc < best_score:
                    best_rows = [list(r) for r in all_rows]
                    best_score = sc
                return True
            
            # Try to build row idx
            row = [0]
            x = 0
            found = False
            while len(row) < k + 1:
                x += 1
                if x > max_allowed:
                    break
                new_diffs = {x - a for a in row}
                if not new_diffs & used_set:
                    row.append(x)
                    used_set = used_set | new_diffs
                    found = True
                    break
                # This x doesn't work, try next
            if not found:
                return False
            
            # For the last row, we need to complete it fully
            # Let's use a more careful approach
            if idx == remaining - 1:
                # Complete the last row carefully
                row = [0]
                x = 0
                used_local = set(used_set)
                while len(row) < k + 1:
                    x += 1
                    if x > max_allowed:
                        return False
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_local:
                        row.append(x)
                        used_local |= new_diffs
                
                all_rows = cur_rows + [row]
                sc = _score(all_rows)
                if sc < best_score:
                    best_rows = [list(r) for r in all_rows]
                    best_score = sc
                return True
            else:
                # For non-last rows, we need to be more careful
                # Just greedily complete and recurse
                row = [0]
                x = 0
                used_local = set(used_set)
                valid = True
                while len(row) < k + 1:
                    x += 1
                    if x > max_allowed:
                        valid = False
                        break
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_local:
                        row.append(x)
                        used_local |= new_diffs
                if valid:
                    if backtrack(idx + 1, used_local, cur_rows + [row], max_allowed):
                        return True
                return False
        
        # Start backtracking
        max_allowed = best_score if best_score < float('inf') else 150
        backtrack(0, used, rows, max_allowed)
        
        # Update max_allowed if we found something better
        if best_score < float('inf'):
            max_allowed = best_score
    
    if best_rows is not None:
        return best_rows
    
    # Fallback: return the greedy solution if we have it
    if len(rows) == n_rows:
        return rows
    
    # Last resort: build as many valid rows as possible, pad with simple rows
    # This shouldn't happen, but just in case
    used = set()
    final_rows = []
    for i in range(n_rows):
        row = _greedy_row(used, k, max_val=500)
        if row is None:
            # Just make a trivial row
            row = list(range(k + 1))
        final_rows.append(row)
        used |= _row_diffs(row)
    return final_rows


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all differences are distinct."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    result = _solve(n_rows, k, deadline=deadline)
    
    # Verify the result is valid
    if result and len(result) == n_rows:
        if _valid_rows(result):
            return result
    
    # Fallback: greedy
    used = set()
    rows = []
    for i in range(n_rows):
        row = _greedy_row(used, k, max_val=500)
        if row is not None:
            rows.append(row)
            used |= _row_diffs(row)
        else:
            # Emergency: just make a row that might be invalid
            rows.append(list(range(k + 1)))
    
    return rows

# EVOLVE-BLOCK-END
