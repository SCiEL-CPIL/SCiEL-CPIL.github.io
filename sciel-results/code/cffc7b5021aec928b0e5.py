import time
import random
import math
import sys
from collections import Counter

_DEADLINE = 0.0
_BEST = None

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _row_diffs(row):
    return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]

def _fast_greedy(n_rows, k, rng):
    """
    Fast greedy initialization using bitmask difference tracking.
    """
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _build_diff_counts(rows):
    """
    Build a Counter of all differences used across all rows.
    """
    counts = Counter()
    for r in rows:
        for d in _row_diffs(r):
            counts[d] += 1
    return counts

def _remove_row_diffs(counts, row):
    for d in _row_diffs(row):
        counts[d] -= 1
        if counts[d] == 0:
            del counts[d]

def _add_row_diffs(counts, row):
    for d in _row_diffs(row):
        counts[d] += 1

def _objective(rows, k):
    """
    Composite objective: scope + sum of all entries.
    Lower is better.
    """
    scope = _scope(rows)
    total = sum(sum(r) for r in rows)
    return scope * 1000 + total

def _perturb_row(row, k, rng, max_scope):
    """
    Randomly perturb a row while keeping it valid (strictly increasing, starts at 0).
    Returns a new row or None if perturbation failed.
    """
    new_row = [0]
    last = 0
    for _ in range(k):
        # Choose a random next value greater than last
        # Bias towards smaller values to encourage tightening
        # But allow some randomness to escape local minima
        min_val = last + 1
        max_val = max_scope - (k - len(new_row))
        
        if min_val > max_val:
            return None
        
        # Random choice with bias towards lower values
        # Use exponential distribution or just uniform
        x = rng.randint(min_val, max_val)
        
        # Ensure no duplicate differences within the row itself
        valid = True
        for a in new_row:
            d = x - a
            if d <= 0:
                valid = False
                break
        if not valid:
            return None
            
        new_row.append(x)
        last = x
        
    return new_row

def _sa_optimize(rows, k, deadline):
    """
    Simulated Annealing optimization.
    """
    rng = random.Random(42)
    
    current_rows = [list(r) for r in rows]
    current_diffs = _build_diff_counts(current_rows)
    current_obj = _objective(current_rows, k)
    current_scope = _scope(current_rows)
    
    best_rows = [list(r) for r in current_rows]
    best_obj = current_obj
    best_scope = current_scope
    
    T = 500.0  # Initial temperature
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with a new random perturbation
            current_rows = [list(r) for r in best_rows]
            current_diffs = _build_diff_counts(current_rows)
            current_obj = _objective(current_rows, k)
            T = 500.0
            # Add some random noise to escape
            for i in range(len(current_rows)):
                new_row = _perturb_row(current_rows[i], k, rng, current_scope + 20)
                if new_row is not None:
                    current_rows[i] = new_row
                    # Check validity
                    if _valid(current_rows, k):
                        current_diffs = _build_diff_counts(current_rows)
                        current_obj = _objective(current_rows, k)
                    else:
                        current_rows[i] = best_rows[i]
                        current_diffs = _build_diff_counts(current_rows)
                        current_obj = _objective(current_rows, k)
        
        # Pick a random row to perturb
        idx = rng.randint(0, len(current_rows) - 1)
        
        # Remove old row diffs
        _remove_row_diffs(current_diffs, current_rows[idx])
        
        # Generate new row
        # Use a more intelligent perturbation: try to keep it close to original
        # but allow random changes
        old_row = current_rows[idx]
        new_row = list(old_row)
        
        # Randomly change one or two elements
        num_changes = rng.randint(1, 2)
        for _ in range(num_changes):
            pos = rng.randint(1, k)  # Don't change the first element (0)
            min_val = new_row[pos-1] + 1
            max_val = new_row[pos+1] - 1 if pos < k else current_scope
            
            if min_val > max_val:
                min_val = max_val = new_row[pos]
            
            new_row[pos] = rng.randint(min_val, max_val)
        
        # Check if new row is strictly increasing
        valid = True
        for i in range(1, len(new_row)):
            if new_row[i] <= new_row[i-1]:
                valid = False
                break
        
        if not valid:
            _add_row_diffs(current_diffs, current_rows[idx])
            T *= cooling
            continue
        
        # Check for conflicts with other rows
        new_diffs = _row_diffs(new_row)
        conflict = False
        for d in new_diffs:
            if d in current_diffs:
                conflict = True
                break
        
        if conflict:
            _add_row_diffs(current_diffs, current_rows[idx])
            T *= cooling
            continue
        
        # Valid move
        _add_row_diffs(current_diffs, new_row)
        new_rows = current_rows
        new_rows[idx] = new_row
        new_obj = _objective(new_rows, k)
        
        # Acceptance criterion
        delta_obj = new_obj - current_obj
        if delta_obj < 0 or (T > 0 and rng.random() < math.exp(-delta_obj / T)):
            current_rows = new_rows
            current_obj = new_obj
            current_scope = _scope(current_rows)
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert
            _remove_row_diffs(current_diffs, new_row)
            _add_row_diffs(current_diffs, current_rows[idx])
        
        T *= cooling
    
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 115.0  # Safety margin
    
    def consider(rows):
        global _BEST
        if rows is None:
            return
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
    
    rng = random.Random(42)
    
    # Phase 1: Greedy Initialization
    best_greedy = None
    best_greedy_scope = float('inf')
    
    for seed in range(20):
        r = random.Random(seed)
        rows = _fast_greedy(n_rows, k, r)
        s = _scope(rows)
        if s < best_greedy_scope:
            best_greedy_scope = s
            best_greedy = rows
            
    if best_greedy is not None:
        consider(best_greedy)
        
    # Phase 2: Simulated Annealing
    if _BEST is not None:
        sa_deadline = _DEADLINE - 5.0
        s, rows = _sa_optimize(_BEST[1], k, sa_deadline)
        consider(rows)
        
    if _BEST is not None:
        return _BEST[1]
    return best_greedy

# Entry point
if __name__ == "__main__":
    # For testing
    rows = construct(7, 5)
    print("Scope:", _scope(rows))
    print("Valid:", _valid(rows, 5))
    for r in rows:
        print(r)
