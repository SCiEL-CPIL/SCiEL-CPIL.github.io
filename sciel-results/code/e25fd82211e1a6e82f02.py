"""Difference triangle set (7,5) - optimized construction."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences a_j - a_i (i<j) within rows must be distinct over all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Phase 1: Interleaved greedy initialization
    rows = _interleaved_greedy(n_rows, k)
    
    # Phase 2: Hill climbing
    rows = _hill_climb(rows, n_rows, k, deadline)
    
    # Phase 3: Simulated annealing with last-element bias
    rows = _simulated_annealing(rows, n_rows, k, deadline)
    
    # Phase 4: Final polish pass
    rows = _final_polish(rows, n_rows, k, deadline)
    
    return rows


def _interleaved_greedy(n_rows, k):
    """Build rows by interleaving: add one element to each row in round-robin fashion.
    This produces a more balanced starting solution than sequential greedy."""
    used = set()
    rows = [[0]] * n_rows
    
    # Each row needs k more elements (k+1 total, starting with 0)
    # Interleave: for each "column" position, pick the smallest valid value for each row
    # Actually, let's do: for each row, greedily pick next element, but process rows in rotation
    # to balance the used differences
    
    # Better approach: build all rows simultaneously, adding one element at a time
    # to the row that currently has the smallest last element
    for _ in range(k):
        # Sort rows by their last element to prioritize shorter rows
        order = sorted(range(n_rows), key=lambda i: rows[i][-1])
        for i in order:
            # Find smallest x > rows[i][-1] such that all new differences are unused
            x = rows[i][-1] + 1
            while True:
                new_diffs = {x - a for a in rows[i]}
                if not new_diffs & used:
                    rows[i].append(x)
                    used |= new_diffs
                    break
                x += 1
    return rows


def _get_scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(row) for row in rows)


def _all_diffs(rows):
    """Return set of all positive differences in all rows."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
    return diffs


def _check_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _row_diffs(row):
    """Return set of all positive differences within a single row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _hill_climb(rows, n_rows, k, deadline):
    """Aggressive hill climbing: try to reduce scope by moving elements."""
    rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    best_rows = [list(r) for r in rows]
    
    while time.time() < deadline - 10:
        improved = False
        # Try moving each element in each row to a smaller valid position
        for ri in range(n_rows):
            for ei in range(1, len(rows[ri])):  # skip index 0 (always 0)
                old_val = rows[ri][ei]
                # Try all values from current-1 down to rows[ri][ei-1]+1
                for new_val in range(old_val - 1, rows[ri][ei-1], -1):
                    if new_val <= rows[ri][ei-1]:
                        break
                    # Check if this move is valid
                    if _is_move_valid(rows, ri, ei, new_val):
                        rows[ri][ei] = new_val
                        new_scope = _get_scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in rows]
                            improved = True
                        # Keep the move even if not better (it might enable further improvements)
                        break  # Take the first valid smaller value
                # Restore if no improvement found at this position
                if not improved:
                    rows[ri][ei] = old_val
        
        if not improved:
            # Try random perturbations
            if time.time() < deadline - 10:
                rows = _perturb(rows, n_rows, k)
                best_rows = [list(r) for r in rows]
                best_scope = _get_scope(rows)
    
    return best_rows


def _is_move_valid(rows, ri, ei, new_val):
    """Check if moving rows[ri][ei] to new_val keeps all differences distinct."""
    row = rows[ri]
    if new_val <= row[ei-1] or new_val >= row[ei+1] if ei+1 < len(row) else False:
        return False
    # Remove old differences involving position ei
    old_diffs = set()
    for j in range(len(row)):
        if j != ei:
            old_diffs.add(abs(row[ei] - row[j]))
    
    # Check new differences
    new_diffs = set()
    for j in range(len(row)):
        if j != ei:
            new_diffs.add(abs(new_val - row[j]))
    
    # The new differences must not conflict with any other row's differences
    # or with this row's other differences
    for d in new_diffs:
        if d in old_diffs:
            continue  # This is fine, it's replacing an old diff
        # Check against all other rows
        for oi in range(len(rows)):
            if oi == ri:
                continue
            if d in _row_diffs(rows[oi]):
                return False
    # Also check new_diffs don't have internal conflicts (shouldn't happen if row stays increasing)
    return True


def _perturb(rows, n_rows, k):
    """Randomly perturb the solution to escape local optima."""
    rows = [list(r) for r in rows]
    # Pick a random row and shift some elements
    for _ in range(random.randint(1, 3)):
        ri = random.randint(0, n_rows - 1)
        ei = random.randint(1, len(rows[ri]) - 1)
        shift = random.randint(1, 5)
        rows[ri][ei] += shift
        # Fix ordering
        for j in range(ei, len(rows[ri]) - 1):
            if rows[ri][j] >= rows[ri][j+1]:
                rows[ri][j+1] = rows[ri][j] + 1
        # Rebuild to ensure validity
        rows = _rebuild_valid(rows, n_rows, k)
    return rows


def _rebuild_valid(rows, n_rows, k):
    """Rebuild rows to ensure validity, keeping structure similar."""
    used = set()
    new_rows = []
    for row in rows:
        new_row = [0]
        for val in row[1:]:
            x = max(val, new_row[-1] + 1)
            while True:
                new_diffs = {x - a for a in new_row}
                if not new_diffs & used:
                    new_row.append(x)
                    used |= new_diffs
                    break
                x += 1
        new_rows.append(new_row)
    return new_rows


def _simulated_annealing(rows, n_rows, k, deadline):
    """SA with bias toward reducing the last (largest) element of each row."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    while time.time() < deadline - 5 and T > min_T:
        # Generate a move with bias toward last elements
        ri = random.randint(0, n_rows - 1)
        # Bias: 70% chance to pick the last element
        if random.random() < 0.7:
            ei = len(rows[ri]) - 1
        else:
            ei = random.randint(1, len(rows[ri]) - 1)
        
        old_val = rows[ri][ei]
        # Try to decrease the value
        delta = random.randint(1, 10)
        new_val = old_val - delta
        min_val = rows[ri][ei-1] + 1
        
        if new_val < min_val:
            new_val = min_val + random.randint(0, 3)
        
        if new_val == old_val:
            continue
        
        if _is_move_valid(rows, ri, ei, new_val):
            rows[ri][ei] = new_val
            new_scope = _get_scope(rows)
            delta_scope = new_scope - best_scope
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
            else:
                # Accept with probability exp(-delta_scope / T)
                if random.random() < _exp_neg(delta_scope / T):
                    pass  # Accept
                else:
                    rows[ri][ei] = old_val  # Reject
        
        T *= cooling
    
    return best_rows


def _exp_neg(x):
    """exp(-x) with clamping to avoid underflow."""
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    import math
    return math.exp(-x)


def _final_polish(rows, n_rows, k, deadline):
    """Final high-precision polish: exhaustive local search on last elements."""
    rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    best_rows = [list(r) for r in rows]
    
    # Multiple passes of trying to reduce each last element
    for _ in range(5):
        if time.time() > deadline - 2:
            break
        for ri in range(n_rows):
            ei = len(rows[ri]) - 1
            old_val = rows[ri][ei]
            # Try all values from old_val-1 down to rows[ri][ei-1]+1
            for new_val in range(old_val - 1, rows[ri][ei-1], -1):
                if _is_move_valid(rows, ri, ei, new_val):
                    rows[ri][ei] = new_val
                    new_scope = _get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    break
    
    # Also try reducing other elements
    for _ in range(3):
        if time.time() > deadline - 2:
            break
        for ri in range(n_rows):
            for ei in range(1, len(rows[ri]) - 1):
                old_val = rows[ri][ei]
                for new_val in range(old_val - 1, rows[ri][ei-1], -1):
                    if _is_move_valid(rows, ri, ei, new_val):
                        rows[ri][ei] = new_val
                        new_scope = _get_scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in rows]
                        break
    
    return best_rows

# EVOLVE-BLOCK-END
