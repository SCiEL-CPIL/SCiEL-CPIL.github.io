"""Construct a difference triangle set (4 rows, k=8) minimizing the largest entry."""

import time
import random
from itertools import combinations

# Precompute difference sets for all 8-subsets of {0,1,...,134} is too large.
# Instead, we'll use a backtracking approach with bitsets for efficiency.

def _diffs_of_row(row):
    """Return the set of all positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    used = set()
    for row in rows:
        diffs = _diffs_of_row(row)
        if diffs & used:
            return False
        used |= diffs
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_build_row(used, max_val):
    """Build a row greedily, picking smallest available integers."""
    row = [0]
    x = 0
    while len(row) < 9:  # k+1 = 9 elements
        x += 1
        if x > max_val:
            return None
        new_diffs = {x - a for a in row}
        if new_diffs & used:
            continue
        row.append(x)
        used |= new_diffs
    return row

def _construct_greedy(n_rows, k, max_val):
    """Greedy construction with a given max_val constraint."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_build_row(used, max_val)
        if row is None:
            return None
        rows.append(row)
    return rows

def _randomized_greedy(n_rows, k, max_val, seed=None):
    """Randomized greedy: shuffle candidate order at each step."""
    if seed is not None:
        random.seed(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return None
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows

def _hill_climb(rows, n_rows, k, max_val, max_steps=50000, deadline=None):
    """Local search: try to reduce max entry by perturbing and repairing."""
    best_rows = [r[:] for r in rows]
    best_scope = _max_entry(best_rows)
    used = set()
    for row in best_rows:
        used |= _diffs_of_row(row)
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        if best_scope <= 112:  # counting bound
            break
            
        # Pick a random row to perturb
        ri = random.randint(0, n_rows - 1)
        old_row = best_rows[ri][:]
        old_diffs = _diffs_of_row(old_row)
        
        # Remove old differences
        used -= old_diffs
        
        # Try to rebuild row ri with a random perturbation
        # Strategy: remove one random element and try to find alternative
        if len(old_row) <= 1:
            best_rows[ri] = old_row
            used |= old_diffs
            continue
            
        # Try shifting one element
        pos = random.randint(0, len(old_row) - 1)
        if pos == 0:
            continue  # keep 0 fixed
        
        orig_val = old_row[pos]
        # Try nearby values
        candidates = []
        for delta in range(-10, 11):
            new_val = orig_val + delta
            if 0 < new_val < max_val and new_val != orig_val:
                # Check strict increase
                if (pos == 0 or new_val > old_row[pos - 1]) and \
                   (pos == len(old_row) - 1 or new_val < old_row[pos + 1]):
                    if new_val not in old_row:
                        candidates.append(new_val)
        
        random.shuffle(candidates)
        found = False
        for new_val in candidates:
            trial_row = old_row[:]
            trial_row[pos] = new_val
            # Ensure strict increase
            valid = True
            for i in range(1, len(trial_row)):
                if trial_row[i] <= trial_row[i-1]:
                    valid = False
                    break
            if not valid:
                continue
            trial_diffs = _diffs_of_row(trial_row)
            if trial_diffs & used:
                continue
            # Accept
            best_rows[ri] = trial_row
            used |= trial_diffs
            new_scope = _max_entry(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
            found = True
            break
        
        if not found:
            # Revert
            best_rows[ri] = old_row
            used |= old_diffs
    
    return best_rows, best_scope

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Initial greedy baseline
    best_rows = _construct_greedy(n_rows, k, 200)
    if best_rows is None:
        best_rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    best_scope = _max_entry(best_rows)
    
    # Try randomized greedy with various seeds
    for attempt in range(100):
        if time.time() > deadline:
            break
        rows = _randomized_greedy(n_rows, k, best_scope, seed=attempt * 1000 + 42)
        if rows is not None:
            scope = _max_entry(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows[:]
    
    # Hill climbing to reduce scope
    for attempt in range(20):
        if time.time() > deadline:
            break
        rows, scope = _hill_climb(best_rows, n_rows, k, best_scope, 
                                   max_steps=20000, deadline=deadline)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # Try to further reduce by trying smaller max_val with greedy
    for target in range(best_scope - 1, max(111, best_scope - 30), -1):
        if time.time() > deadline:
            break
        for seed in range(5):
            rows = _randomized_greedy(n_rows, k, target, seed=seed * 777 + 123)
            if rows is not None:
                scope = _max_entry(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows[:]
                    break
    
    return best_rows
