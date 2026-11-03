"""Improved program using structured construction + simulated annealing for difference triangle sets."""
import time
import random
import math
from collections import set

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimize the largest entry (scope)."""
    
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Counting bound: scope must be at least total_diffs
    min_scope = total_diffs
    
    # Best known for (7,5) is 112
    # Start with a good initial solution using greedy
    best_rows = _greedy_construct(n_rows, k)
    best_scope = _scope(best_rows)
    
    if best_rows is None or not _valid(best_rows, n_rows, k):
        # Fallback: try harder greedy
        best_rows = _greedy_construct_hard(n_rows, k)
        best_scope = _scope(best_rows)
    
    # Now use simulated annealing to improve
    if time.time() < deadline - 5.0:
        improved = _sa_optimize(best_rows, n_rows, k, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_rows = improved
                best_scope = imp_scope
    
    # Final validation
    if best_rows is not None and _valid(best_rows, n_rows, k):
        return best_rows
    else:
        # Return whatever we have, even if not optimal
        return best_rows if best_rows else [[0]] * n_rows


def _scope(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(row) for row in rows)


def _valid(rows, n_rows, k):
    """Validate that the rows satisfy all constraints."""
    if rows is None or len(rows) != n_rows:
        return False
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True


def _diffs_for_row(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _build_all_diffs(rows):
    """Return the set of all differences across all rows."""
    all_diffs = set()
    for row in rows:
        all_diffs |= _diffs_for_row(row)
    return all_diffs


def _greedy_construct(n_rows, k):
    """Greedy construction: pick smallest valid entry at each step."""
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


def _greedy_construct_hard(n_rows, k):
    """Harder greedy: try multiple starting patterns."""
    best = None
    best_scope = float('inf')
    
    # Try different initial steps
    for first_step in range(1, 6):
        used = set()
        rows = []
        valid = True
        for r in range(n_rows):
            row = [0]
            if r == 0:
                # First row: use first_step as initial increment
                x = first_step
            else:
                x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                else:
                    if x > 200:
                        valid = False
                        break
            if not valid:
                break
            rows.append(row)
        
        if valid and len(rows) == n_rows:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best = rows
    
    if best is None:
        best = _greedy_construct(n_rows, k)
    
    return best


def _sa_optimize(rows, n_rows, k, deadline):
    """Simulated annealing to minimize scope."""
    if rows is None:
        return None
    
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    best = [list(r) for r in current]
    best_scope = current_scope
    
    # Temperature schedule
    T = current_scope * 0.5
    T_min = 0.1
    cooling = 0.995
    
    rng = random.Random(42)
    
    while time.time() < deadline - 3.0 and T > T_min:
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Move: change one entry in a random row
            new_rows = [list(r) for r in current]
            ri = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # positions 1..k (not position 0)
            
            # Try to reduce the entry (scope reduction bias)
            if rng.random() < 0.7:
                # Try smaller values
                old_val = new_rows[ri][pos]
                for delta in range(1, min(old_val, 10) + 1):
                    new_val = old_val - delta
                    if new_val <= new_rows[ri][pos - 1]:
                        continue
                    # Check validity
                    trial_row = new_rows[ri][:]
                    trial_row[pos] = new_val
                    if _row_valid(trial_row, pos):
                        # Check global validity
                        all_diffs = _build_all_diffs(new_rows)
                        # Remove old row's diffs, add new
                        old_diffs = _diffs_for_row(new_rows[ri])
                        new_diffs = _diffs_for_row(trial_row)
                        all_diffs -= old_diffs
                        if new_diffs & all_diffs:
                            continue
                        all_diffs |= new_diffs
                        new_rows[ri] = trial_row
                        break
            else:
                # Try larger values (to escape local minima)
                old_val = new_rows[ri][pos]
                new_val = old_val + rng.randint(1, 5)
                if pos == k or new_val < new_rows[ri][pos + 1] if pos + 1 <= k else True:
                    trial_row = new_rows[ri][:]
                    trial_row[pos] = new_val
                    if _row_valid(trial_row, pos):
                        old_diffs = _diffs_for_row(new_rows[ri])
                        new_diffs = _diffs_for_row(trial_row)
                        all_diffs = _build_all_diffs(new_rows)
                        all_diffs -= old_diffs
                        if new_diffs & all_diffs:
                            continue
                        all_diffs |= new_diffs
                        new_rows[ri] = trial_row
        
        elif move_type < 0.7:
            # Move: swap two entries in the same row
            new_rows = [list(r) for r in current]
            ri = rng.randint(0, n_rows - 1)
            p1, p2 = rng.sample(range(1, k + 1), 2)
            if p1 > p2:
                p1, p2 = p2, p1
            # Swap values
            new_rows[ri][p1], new_rows[ri][p2] = new_rows[ri][p2], new_rows[ri][p1]
            # Check if still increasing
            if not _row_valid(new_rows[ri], -1):
                new_rows = [list(r) for r in current]
                continue
            # Check global
            old_diffs = _diffs_for_row(current[ri])
            new_diffs = _diffs_for_row(new_rows[ri])
            all_diffs = _build_all_diffs(current)
            all_diffs -= old_diffs
            if new_diffs & all_diffs:
                new_rows = [list(r) for r in current]
                continue
        
        elif move_type < 0.85:
            # Move: compress a row by shifting entries down
            new_rows = [list(r) for r in current]
            ri = rng.randint(0, n_rows - 1)
            # Try to reduce the last entry
            old_last = new_rows[ri][k]
            for delta in range(1, min(old_last, 5) + 1):
                new_last = old_last - delta
                if new_last <= new_rows[ri][k - 1]:
                    continue
                trial_row = new_rows[ri][:]
                trial_row[k] = new_last
                if _row_valid(trial_row, k):
                    old_diffs = _diffs_for_row(new_rows[ri])
                    new_diffs = _diffs_for_row(trial_row)
                    all_diffs = _build_all_diffs(new_rows)
                    all_diffs -= old_diffs
                    if new_diffs & all_diffs:
                        continue
                    new_rows[ri] = trial_row
                    break
        
        else:
            # Move: perturb a random row more aggressively
            new_rows = [list(r) for r in current]
            ri = rng.randint(0, n_rows - 1)
            # Regenerate this row with slight perturbation
            base = current[ri]
            new_row = [0]
            for pos in range(1, k + 1):
                base_val = base[pos] if pos < len(base) else base[-1] + pos
                new_val = max(base_val + rng.randint(-3, 3), new_row[-1] + 1)
                new_row.append(new_val)
            # Check validity
            if _row_valid(new_row, -1):
                old_diffs = _diffs_for_row(new_rows[ri])
                new_diffs = _diffs_for_row(new_row)
                all_diffs = _build_all_diffs(new_rows)
                all_diffs -= old_diffs
                if not (new_diffs & all_diffs):
                    new_rows[ri] = new_row
                else:
                    new_rows = [list(r) for r in current]
        
        # Evaluate
        new_scope = _scope(new_rows)
        delta = new_scope - current_scope
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
            current = new_rows
            current_scope = new_scope
            if new_scope < best_scope:
                best = [list(r) for r in current]
                best_scope = new_scope
        
        T *= cooling
    
    return best


def _row_valid(row, check_pos):
    """Check if a row is strictly increasing."""
    for i in range(len(row) - 1):
        if row[i + 1] <= row[i]:
            return False
    return True
