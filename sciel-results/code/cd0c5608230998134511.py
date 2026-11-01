"""Rewritten program: difference triangle set (4,7) via randomized local search."""
import time
import random
import math
import sys

# Deadline guard: evaluator kills at 150s wall-clock; use 145s limit for safety.
_DEADLINE = 145.0

def _elapsed_ok():
    return time.time() < _DEADLINE

def _row_diffs(row):
    """Return the set of positive differences within a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_diffs(rows):
    """Return the union of all positive differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used

def _is_valid(rows, n_rows, k):
    """Check that all rows start at 0, are strictly increasing, have k+1 elements,
    and all positive differences are distinct across all rows."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & used:
            return False
        used |= rd
    return True

def _scope(rows):
    """The largest entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, rng=None):
    """Greedy: each row takes the smallest next mark whose differences are unused."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _random_row(used, k, rng, max_val_hint):
    """Generate a random row of k+1 increasing integers starting at 0,
    avoiding used differences. Returns None if impossible."""
    row = [0]
    remaining = k
    # Estimate how large we need to go
    # Each new element adds (len(row)) new differences
    # We need total k(k+1)/2 differences per row, all distinct from used
    # Use a simple randomized approach
    for _ in range(remaining):
        found = False
        # Try random positions in increasing order
        # The new element must be > row[-1]
        # And all differences (x - a for a in row) must not be in used
        # We'll search up to some bound
        upper = max_val_hint + 50
        for attempt in range(200):
            if row:
                lo = row[-1] + 1
            else:
                lo = 1
            if attempt == 0:
                x = lo
            else:
                x = rng.randint(lo, min(upper, lo + 300))
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used = used | new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _local_search(rows, n_rows, k, rng, time_limit=140.0):
    """Local search: perturb one row at a time, keep if scope decreases or equal."""
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    start = time.time()
    
    # Precompute
    k1 = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    
    while time.time() - start < time_limit:
        # Pick a row to modify
        row_idx = rng.randrange(n_rows)
        
        # Get all differences used by other rows
        other_used = set()
        for i, r in enumerate(best_rows):
            if i != row_idx:
                other_used |= _row_diffs(r)
        
        # Try to rebuild row_idx with a target max value
        # Start from current max and try to reduce
        current_max = best_rows[row_idx][-1]
        # Try a few target maxima
        found_better = False
        for target_max in range(max(1, current_max - 1), max(0, current_max - 200), -1):
            if time.time() - start > time_limit:
                break
            # Try to construct a row with max <= target_max
            # Using randomized backtracking
            row = _construct_row_limited(other_used, k, target_max, rng, max_attempts=500)
            if row is not None:
                new_scope = max(max(r) for r in best_rows) if row_idx != 0 else max(r[-1] for r in best_rows)
                # Actually compute scope properly
                new_rows = list(best_rows)
                new_rows[row_idx] = row
                ns = _scope(new_rows)
                if ns < best_scope:
                    best_rows = new_rows
                    best_scope = ns
                    found_better = True
                    break
                elif ns == best_scope:
                    # Accept with some probability to allow exploration
                    if rng.random() < 0.3:
                        best_rows = new_rows
                        found_better = True
                        break
        # If no improvement, try a random perturbation
        if not found_better and rng.random() < 0.5:
            # Perturb: shift some elements
            new_row = list(best_rows[row_idx])
            # Try to reduce the last element
            if len(new_row) > 1:
                # Try reducing the last element
                new_row[-1] = new_row[-1] - rng.randint(1, 5)
                # Ensure strictly increasing
                valid = True
                for i in range(len(new_row) - 1):
                    if new_row[i + 1] <= new_row[i]:
                        valid = False
                        break
                if valid and new_row[0] == 0:
                    # Check differences
                    rd = _row_diffs(new_row)
                    other_used = set()
                    for i, r in enumerate(best_rows):
                        if i != row_idx:
                            other_used |= _row_diffs(r)
                    if not rd & other_used:
                        new_rows = list(best_rows)
                        new_rows[row_idx] = new_row
                        ns = _scope(new_rows)
                        if ns <= best_scope:
                            best_rows = new_rows
                            best_scope = ns
    return best_rows, best_scope

def _construct_row_limited(used, k, max_val, rng, max_attempts=500):
    """Try to construct a row of k+1 elements starting at 0, all <= max_val,
    with all differences not in 'used'. Returns the row or None."""
    k1 = k + 1
    
    # Backtracking with randomization
    def backtrack(row, start_pos):
        if len(row) == k1:
            return row
        # Remaining elements needed
        remaining = k1 - len(row)
        # The last element must be <= max_val
        # We need 'remaining' more elements, each at least 1 apart
        # So the minimum possible last element is row[-1] + remaining
        if row[-1] + remaining > max_val:
            return None
        
        # Try positions from row[-1]+1 to max_val
        # Randomize the order somewhat
        lo = row[-1] + 1
        hi = max_val
        if hi < lo:
            return None
        
        # Limit search space: try a subset of positions
        # The new element x must satisfy: x - a not in used for all a in row
        # And x - a must be >= 1 (automatic since x > a)
        # Also x - a must be <= max_val (automatic)
        
        # Collect valid candidates
        candidates = []
        # To limit work, sample positions
        span = hi - lo + 1
        if span <= 100:
            candidates = list(range(lo, hi + 1))
            rng.shuffle(candidates)
        else:
            # Sample positions
            num_samples = min(200, span)
            candidates = rng.sample(range(lo, hi + 1), num_samples)
        
        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            # Check that remaining elements can fit
            # We need k1 - len(row) - 1 more elements after x
            # Each at least x+1, so the max possible is x + (k1 - len(row) - 1)
            if x + (k1 - len(row) - 1) > max_val and len(row) + 1 < k1:
                continue
            
            result = backtrack(row + [x], start_pos)
            if result is not None:
                return result
        return None
    
    # Try multiple times with different random seeds
    for _ in range(max_attempts):
        row = backtrack([0], 0)
        if row is not None:
            return row
    return None

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences are distinct."""
    start_time = time.time()
    deadline = start_time + _DEADLINE - 2.0  # Safety margin
    
    # Initialize with greedy
    rng = random.Random(42)
    rows = _greedy_construct(n_rows, k, rng)
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    # Local search
    if time.time() < deadline:
        rows2, scope2 = _local_search(rows, n_rows, k, rng, time_limit=min(100.0, deadline - time.time()))
        if scope2 < best_scope:
            best_rows = rows2
            best_scope = scope2
    
    # Try a few different random initializations
    for seed in range(50):
        if time.time() > deadline:
            break
        rng2 = random.Random(seed * 1000 + 7)
        # Randomized greedy: sometimes skip the smallest available
        used = set()
        rows_init = []
        ok = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    # With some probability, skip this and try next
                    if rng2.random() < 0.1 and x < best_scope:
                        continue
                    row.append(x)
                    used |= new
            rows_init.append(row)
        s = _scope(rows_init)
        if s < best_scope:
            best_rows = [list(r) for r in rows_init]
            best_scope = s
        
        # Quick local search from this start
        if time.time() < deadline - 5.0 and s <= best_scope:
            rows3, scope3 = _local_search(rows_init, n_rows, k, rng2, time_limit=min(20.0, deadline - time.time() - 5.0))
            if scope3 < best_scope:
                best_rows = rows3
                best_scope = scope3
    
    # Final validation
    if not _is_valid(best_rows, n_rows, k):
        # Fallback to greedy
        best_rows = _greedy_construct(n_rows, k)
    
    return best_rows
