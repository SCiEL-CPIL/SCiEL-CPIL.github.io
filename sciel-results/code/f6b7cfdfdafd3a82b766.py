"""Stronger algorithm for difference triangle set (7,5) via backtracking with restarts."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _backtrack_row(used_diffs, k, max_val, row, depth):
    """
    Backtrack to build one row of length k+1 starting with 0.
    Returns the row if found, else None.
    """
    if depth == k + 1:
        return row[:]
    
    # Try values from row[-1]+1 to max_val
    start = row[-1] + 1
    for x in range(start, max_val + 1):
        new_diffs = {x - a for a in row}
        if new_diffs & used_diffs:
            continue
        row.append(x)
        result = _backtrack_row(used_diffs, k, max_val, row, depth + 1)
        if result is not None:
            return result
        row.pop()
    return None

def _backtrack_rows(used_diffs, n_rows, k, max_val, rows, row_idx):
    """
    Backtrack to build all n_rows rows with max entry <= max_val.
    Returns list of rows if found, else None.
    """
    if row_idx == n_rows:
        return rows[:]
    
    # Build the next row
    row = [0]
    result = _backtrack_row(used_diffs, k, max_val, row, 1)
    if result is None:
        return None
    
    new_diffs = _row_diffs(result)
    rows.append(result)
    result = _backtrack_rows(used_diffs, n_rows, k, max_val, rows, row_idx + 1)
    if result is not None:
        return result
    rows.pop()
    return None

def _try_max_val(n_rows, k, max_val, time_limit, rng):
    """
    Try to find a valid configuration with max entry = max_val.
    Uses randomized backtracking with restarts.
    Returns list of rows if found, else None.
    """
    deadline = time.time() + time_limit
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    if max_val < total_diffs:
        return None  # Not enough room for all differences
    
    best_result = None
    restarts = 0
    max_restarts = 200
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        used_diffs = set()
        rows = []
        
        # Randomized backtracking: shuffle the order we try values
        # We'll use a simple approach: try backtracking, and if it fails,
        # we restart with different random choices
        
        # For the first attempt, try deterministic (smallest values first)
        if restarts == 1:
            result = _backtrack_rows(used_diffs, n_rows, k, max_val, rows, 0)
            if result is not None:
                return result
        else:
            # Randomized: we can't easily randomize the backtracking without
            # major restructuring, so we'll try a different approach:
            # Build rows with some randomization in value selection
            result = _randomized_build_rows(n_rows, k, max_val, time.time(), deadline, rng)
            if result is not None:
                return result
    
    return best_result

def _randomized_build_rows(n_rows, k, max_val, start_time, deadline, rng):
    """
    Build rows with randomized value selection to escape local minima.
    """
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        row = [0]
        # Build the row with some randomization
        success = _build_row_randomized(used_diffs, k, max_val, row, 1, rng, deadline)
        if not success:
            return None
        
        new_diffs = _row_diffs(row)
        rows.append(row)
        used_diffs |= new_diffs
    
    return rows

def _build_row_randomized(used_diffs, k, max_val, row, depth, rng, deadline):
    """
    Build one row with randomized backtracking.
    """
    if time.time() > deadline:
        return False
    
    if depth == k + 1:
        return True
    
    start = row[-1] + 1
    if start > max_val:
        return False
    
    # Generate candidate values
    candidates = list(range(start, max_val + 1))
    
    # With some probability, shuffle to introduce randomness
    if rng.random() < 0.3:
        rng.shuffle(candidates)
    else:
        # Sort by a heuristic: prefer values that create "rare" differences
        # (differences not yet used)
        def heuristic(x):
            new_diffs = {x - a for a in row}
            # Count how many of these diffs are already used
            conflicts = len(new_diffs & used_diffs)
            return conflicts
        
        candidates.sort(key=heuristic)
    
    for x in candidates:
        if time.time() > deadline:
            return False
        
        new_diffs = {x - a for a in row}
        if new_diffs & used_diffs:
            continue
        
        row.append(x)
        if _build_row_randomized(used_diffs, k, max_val, row, depth + 1, rng, deadline):
            return True
        row.pop()
    
    return False

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are globally distinct.
    Minimize the largest entry.
    """
    start_time = time.time()
    total_time_limit = 140  # Leave 10s safety margin for 150s limit
    
    # Lower bound: we need n_rows * C(k+1, 2) distinct differences in [1, M]
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # The counting bound: M >= total_diffs (since differences are in 1..M)
    lower_bound = total_diffs
    
    # Try from lower_bound upward
    # For (7,5): lower_bound = 7*21 = 147... wait, that's not right.
    # Differences are in [1, M], and we need total_diffs distinct ones.
    # So M >= total_diffs. For 7 rows, 21 diffs each: 147. But known best is 112.
    # Ah, the counting bound mentioned is 105, which suggests a tighter analysis.
    # Actually, the differences don't all have to be distinct integers 1..M;
    # they just need to be distinct. The minimum possible max difference is
    # constrained by the fact that the largest element is M, so all differences
    # are in [1, M]. We need 147 distinct values in [1, M], so M >= 147.
    # But the problem says "simple counting bound is 105" - this must be a
    # different counting argument.
    
    # Let me re-examine: for k=5 (6 elements per row? No, k=5 means k+1=6 elements)
    # Wait, the task says "7 rows of 6 increasing integers" and "k" is passed.
    # If k=5, then k+1=6 elements. C(6,2)=15 differences per row. 7*15=105.
    # So the counting bound is 105. That makes sense!
    
    # So for the general case: lower_bound = n_rows * k * (k+1) // 2
    
    # But wait, the problem statement says "7 rows of 6 increasing integers starting at 0"
    # and the interface is construct(n_rows, k). If k=5, we have 6 elements.
    # C(6,2) = 15. 7*15 = 105. Yes!
    
    # Let me recalculate: diffs_per_row should be C(k+1, 2) = (k+1)*k//2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Try M from lower_bound upward
    rng = random.Random(42)
    
    best_rows = None
    best_max_val = float('inf')
    
    # First, try the greedy approach to get a baseline
    greedy_rows = _greedy_construct(n_rows, k)
    if greedy_rows:
        best_rows = greedy_rows
        best_max_val = max(max(r) for r in greedy_rows)
    
    # Now try backtracking from lower_bound up to best_max_val - 1
    for M in range(lower_bound, best_max_val):
        if time.time() - start_time > total_time_limit - 10:
            break
        
        time_for_this_M = min(10.0, total_time_limit - (time.time() - start_time) - 5)
        if time_for_this_M <= 0:
            break
        
        result = _try_max_val(n_rows, k, M, time_for_this_M, rng)
        if result is not None:
            best_rows = result
            best_max_val = M
            break  # Found the minimum, we're done
    
    # If we didn't find anything better, return the greedy result
    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k)
    
    return best_rows

def _greedy_construct(n_rows, k):
    """Greedy construction as baseline."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
