"""Construct difference triangle sets via randomized greedy + local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        d = _row_diffs(row)
        if d & seen:
            return False
        seen |= d
    return True

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_row(used, k, rng):
    """Build one row of k+1 increasing ints starting at 0, greedily.
    Returns (row, new_diffs) or None if stuck (shouldn't happen with enough space)."""
    row = [0]
    row_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & row_diffs):
            row.append(x)
            row_diffs |= new
    return row, row_diffs

def _greedy_all(n_rows, k, rng):
    """Build all rows greedily in random order."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _greedy_row(used, k, rng)
        rows.append(row)
        used |= diffs
    return rows

def _local_search(rows, k, rng, time_limit=5.0):
    """Try to reduce scope by moving individual elements to smaller valid positions.
    Returns improved rows or original if no improvement / time up."""
    start = time.time()
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    n_rows = len(rows)
    n_per_row = k + 1
    
    # Precompute all differences for quick check
    def all_diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                s.add(row[j] - row[i])
        return s
    
    # We'll try to reduce the maximum element.
    # Strategy: for each row, try to shift the last element down, then rebuild.
    # More effective: try to reduce any element while maintaining distinctness.
    
    # Simpler approach: repeatedly pick a random row and try to compress it
    # by trying smaller values for its elements (from the end backwards).
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try each row
        for ri in range(n_rows):
            if (time.time() - start) > time_limit:
                break
            row = list(best[ri])
            # Try to reduce each element from the last one backwards
            for pos in range(k, 0, -1):
                if (time.time() - start) > time_limit:
                    break
                orig_val = row[pos]
                # Try smaller values
                for new_val in range(orig_val - 1, row[pos-1], -1):
                    row[pos] = new_val
                    # Check validity: differences within this row must be distinct
                    # and not conflict with other rows
                    row_d = all_diffs(row)
                    # Check internal distinctness (already guaranteed by construction if increasing)
                    # Check against all other rows
                    conflict = False
                    for rj in range(n_rows):
                        if rj == ri:
                            continue
                        other_d = all_diffs(best[rj])
                        if row_d & other_d:
                            conflict = True
                            break
                    if not conflict:
                        best[ri] = row
                        new_scope = _scope(best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        improved = True
                        break
                else:
                    # No change made at this position, restore
                    row[pos] = orig_val
                    continue
                # If we made a change, break out of pos loop for this row
                break
    return best

def _randomized_greedy(n_rows, k, rng, time_limit=30.0):
    """Run randomized greedy with multiple restarts, keeping best."""
    start = time.time()
    best_rows = None
    best_scope = float('inf')
    
    attempt = 0
    while (time.time() - start) < time_limit:
        attempt += 1
        rows = _greedy_all(n_rows, k, rng)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
        # Local search to improve
        rows_ls = _local_search(rows, k, rng, time_limit=2.0)
        scope_ls = _scope(rows_ls)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = [list(r) for r in rows_ls]
    
    return best_rows

def _hill_climbing(n_rows, k, rng, time_limit=60.0):
    """Hill-climbing: start from a greedy solution, try random perturbations."""
    start = time.time()
    # Start with a good initial solution
    best_rows = _greedy_all(n_rows, k, rng)
    best_scope = _scope(best_rows)
    
    # Local search first
    best_rows = _local_search(best_rows, k, rng, time_limit=5.0)
    best_scope = _scope(best_rows)
    
    n_per_row = k + 1
    max_val = best_scope
    
    while (time.time() - start) < time_limit:
        # Pick a random row and a random position to perturb
        ri = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)  # Don't perturb position 0 (always 0)
        
        # Try a few random new values
        for _ in range(10):
            new_val = rng.randint(best_rows[ri][pos-1] + 1, max(1, best_scope))
            if new_val == best_rows[ri][pos]:
                continue
            # Build candidate row
            cand = list(best_rows[ri])
            cand[pos] = new_val
            # Check if still strictly increasing
            if any(cand[i] >= cand[i+1] for i in range(pos-1, pos)):
                continue
            if any(cand[i] >= cand[i+1] for i in range(pos, min(pos+1, n_per_row-1))):
                continue
            
            # Check validity
            cand_d = _row_diffs(cand)
            conflict = False
            for rj in range(n_rows):
                if rj == ri:
                    continue
                other_d = _row_diffs(best_rows[rj])
                if cand_d & other_d:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Accept
            best_rows[ri] = cand
            new_scope = _scope(best_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                # Run local search
                best_rows = _local_search(best_rows, k, rng, time_limit=2.0)
                best_scope = _scope(best_rows)
            break
    
    return best_rows

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    with all positive differences distinct across all rows.
    Minimizes the largest entry."""
    deadline = time.time() + 110  # Leave 10s safety margin
    
    rng = random.Random(42)
    
    # Phase 1: Randomized greedy with multiple restarts
    best_rows = _randomized_greedy(n_rows, k, rng, time_limit=min(30.0, (deadline - time.time()) * 0.3))
    
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = _greedy_all(n_rows, k, rng)
    
    # Phase 2: Hill climbing / local search
    remaining = deadline - time.time()
    if remaining > 10:
        best_rows = _hill_climbing(n_rows, k, rng, time_limit=min(remaining - 10, 60.0))
    
    # Final local search polish
    remaining = deadline - time.time()
    if remaining > 5:
        best_rows = _local_search(best_rows, k, rng, time_limit=min(remaining - 5, 10.0))
    
    # Verify validity
    if not _is_valid(best_rows):
        # Shouldn't happen, but fallback to simple greedy
        best_rows = _greedy_all(n_rows, k, rng)
    
    return best_rows

# EVOLVE-BLOCK-END
