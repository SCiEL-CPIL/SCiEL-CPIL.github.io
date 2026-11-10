"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _all_diffs(row):
    """Return the set of all positive differences in a single row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _validate(rows):
    """Check that all positive differences across all rows are distinct."""
    used = set()
    for row in rows:
        d = _all_diffs(row)
        if d & used:
            return False
        used |= d
    return True


def _scope(rows):
    """Return the maximum entry (scope) over all rows."""
    return max(max(r) for r in rows)


def _greedy_row(used, k, max_val, rng):
    """Build one row greedily: pick smallest x such that new diffs don't collide.
    Returns the row list or None if impossible within max_val."""
    row = [0]
    x = 1
    while len(row) < k + 1:
        if x > max_val:
            return None
        new = set()
        ok = True
        for a in row:
            d = x - a
            if d in used:
                ok = False
                break
            new.add(d)
        if ok:
            row.append(x)
            used |= new
            x += 1
        else:
            x += 1
    return row


def _random_greedy_row(used, k, max_val, rng):
    """Build a row with some randomness: at each step, pick from a small window
    of candidate x values (not just the smallest) to diversify."""
    row = [0]
    x = 1
    window = 3  # consider up to 'window' candidates
    while len(row) < k + 1:
        candidates = []
        for _ in range(window):
            if x > max_val:
                break
            new = set()
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
            if ok:
                candidates.append((x, new))
            x += 1
        if not candidates:
            return None
        # Pick randomly among candidates (slightly biased toward smaller)
        idx = rng.randint(0, min(1, len(candidates) - 1))
        chosen_x, new_diffs = candidates[idx]
        row.append(chosen_x)
        used |= new_diffs
        # Next search starts after chosen_x
        x = chosen_x + 1
    return row


def _local_search(rows, k, rng, time_limit):
    """Local search: try to reduce scope by shifting elements or replacing rows.
    Returns improved rows or original if no improvement found."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    start = time.time()
    
    # Precompute all differences
    def get_all_diffs(rows):
        s = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    s.add(row[j] - row[i])
        return s
    
    current_diffs = get_all_diffs(best_rows)
    
    # Element shift move: pick a row, pick a position, shift it by +/-1
    # Check if still increasing and no diff collision
    while time.time() - start < time_limit:
        improved = False
        n_rows = len(best_rows)
        for _ in range(50):  # limit iterations per outer loop
            if time.time() - start > time_limit:
                break
            r_idx = rng.randint(0, n_rows - 1)
            row = best_rows[r_idx]
            pos = rng.randint(1, k)  # positions 1..k (not 0)
            delta = rng.choice([-1, 1])
            new_val = row[pos] + delta
            
            # Check increasing
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < k and new_val >= row[pos + 1]:
                continue
            if new_val < 0:
                continue
            
            # Compute old diffs involving this element
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(row[pos] - row[i])
            for j in range(pos + 1, k + 1):
                old_diffs.add(row[j] - row[pos])
            
            # Compute new diffs
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, k + 1):
                new_diffs.add(row[j] - new_val)
            
            # Check collisions: new_diffs must not be in (current_diffs - old_diffs)
            other_diffs = current_diffs - old_diffs
            if new_diffs & other_diffs:
                continue
            
            # Also check new_diffs doesn't collide with itself (shouldn't for single element)
            # Apply
            row[pos] = new_val
            current_diffs = (current_diffs - old_diffs) | new_diffs
            improved = True
            
            # Check scope
            new_scope = _scope(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
        
        if not improved and time.time() - start > time_limit * 0.5:
            break
    
    return best_rows, best_scope


def _solve_greedy_with_restarts(n_rows, k, max_val, n_restarts, rng, time_limit):
    """Try multiple greedy restarts, keep best scope."""
    best_rows = None
    best_scope = float('inf')
    start = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start > time_limit:
            break
        used = set()
        rows = []
        success = True
        for i in range(n_rows):
            if time.time() - start > time_limit:
                success = False
                break
            if i % 3 == 0:
                row = _greedy_row(used, k, max_val, rng)
            else:
                row = _random_greedy_row(used, k, max_val, rng)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= _all_diffs(row)
        
        if success:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    return best_rows, best_scope


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope) subject to all positive differences
    being distinct across all rows."""
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Phase 1: Pure greedy to get a baseline
    used = set()
    baseline_rows = []
    for i in range(n_rows):
        row = _greedy_row(used, k, 10000, rng)
        if row is None:
            # Fallback: simple greedy
            row = [0]
            x = 1
            while len(row) < k + 1:
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                x += 1
            baseline_rows.append(row)
        else:
            baseline_rows.append(row)
            used |= _all_diffs(row)
    
    best_rows = [list(r) for r in baseline_rows]
    best_scope = _scope(best_rows)
    
    # Phase 2: Randomized greedy restarts
    elapsed = time.time() - deadline
    remaining = 140.0 - (time.time() - (deadline - 140.0))
    # Better: compute remaining time
    start_time = deadline - 140.0
    remaining = max(0, 140.0 - (time.time() - start_time))
    
    if remaining > 20:
        # Estimate max_val from baseline
        max_val = int(best_scope * 1.2) + 10
        n_restarts = min(50, max(5, int(remaining * 2)))
        rows2, scope2 = _solve_greedy_with_restarts(n_rows, k, max_val, n_restarts, rng, remaining * 0.5)
        if rows2 is not None and scope2 < best_scope:
            best_rows = rows2
            best_scope = scope2
    
    # Phase 3: Local search on best solution
    remaining = max(0, 140.0 - (time.time() - start_time))
    if remaining > 10:
        improved_rows, improved_scope = _local_search(
            best_rows, k, rng, remaining * 0.7
        )
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
    
    # Final validation
    if not _validate(best_rows):
        # Fallback to baseline
        best_rows = [list(r) for r in baseline_rows]
    
    return best_rows

# EVOLVE-BLOCK-END
