"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate(rows):
    """Check that all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        diffs = _row_diffs(row)
        if diffs & seen:
            return False
        seen |= diffs
    return True

def _scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_row(used, k, rng, strategy=0):
    """Build one row greedily with randomization options."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        if strategy == 0:
            # Pure greedy: smallest available
            x += 1
            while True:
                new = {x - a for a in row}
                if not new & used:
                    break
                x += 1
            row.append(x)
            used |= new
        elif strategy == 1:
            # Randomized greedy: try a random window of candidates
            best_x = None
            for _ in range(5):
                trial = x + 1 + rng.randint(0, 10)
                new = {trial - a for a in row}
                if not new & used:
                    best_x = trial
                    break
            if best_x is None:
                # Fallback to sequential
                trial = x + 1
                while True:
                    new = {trial - a for a in row}
                    if not new & used:
                        best_x = trial
                        break
                    trial += 1
            x = best_x
            row.append(x)
            used |= new
        else:
            # Backtracking-lite: try several candidates, pick one that leaves most room
            candidates = []
            trial = x + 1
            for _ in range(20):
                new = {trial - a for a in row}
                if not new & used:
                    candidates.append(trial)
                    if len(candidates) >= 3:
                        break
                trial += 1
            if not candidates:
                # Fallback
                trial = x + 1
                while True:
                    new = {trial - a for a in row}
                    if not new & used:
                        candidates.append(trial)
                        break
                    trial += 1
            x = candidates[0]
            row.append(x)
            used |= new
    return row

def _construct_greedy(n_rows, k, rng, strategy=0):
    """Construct all rows using a greedy approach with given strategy."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, strategy)
        rows.append(row)
    return rows

def _local_improve(rows, n_rows, k, used, rng, deadline):
    """Try to improve by rebuilding individual rows with randomization."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        # Pick a random row to rebuild
        idx = rng.randint(0, n_rows - 1)
        old_row = best_rows[idx]
        old_diffs = _row_diffs(old_row)
        
        # Remove this row's differences from used
        new_used = set()
        for i, row in enumerate(best_rows):
            if i != idx:
                new_used |= _row_diffs(row)
        
        # Try to build a new row with smaller max
        target_max = old_row[-1] - 1
        if target_max < k:  # minimum possible max for k+1 elements starting at 0
            continue
        
        found = False
        for _ in range(10):
            row = [0]
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                if x > target_max:
                    success = False
                    break
                new = {x - a for a in row}
                if new & new_used:
                    continue
                row.append(x)
            if success and len(row) == k + 1:
                # Check if this is better
                if row[-1] < old_row[-1]:
                    best_rows[idx] = row
                    best_scope = _scope(best_rows)
                    found = True
                    break
        
    return best_rows, best_scope

def construct(n_rows, k):
    """Main entry point: find difference triangle set with minimal scope."""
    deadline = time.time() + 140  # 140s safety margin
    
    rng = random.Random(42)
    
    # Multiple strategies for initial construction
    strategies = [0, 1, 2]
    seeds = [42, 123, 7, 999, 31415]
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try multiple greedy constructions
    for strategy in strategies:
        for seed in seeds:
            if time.time() > deadline:
                break
            rng2 = random.Random(seed + strategy * 1000)
            rows = _construct_greedy(n_rows, k, rng2, strategy)
            if _validate(rows):
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
        if time.time() > deadline:
            break
    
    # Phase 2: Local improvement on best solution
    if best_rows is not None:
        if time.time() < deadline:
            used = set()
            for row in best_rows:
                used |= _row_diffs(row)
            improved_rows, improved_scope = _local_improve(
                best_rows, n_rows, k, used, rng, deadline
            )
            if improved_scope < best_scope:
                best_rows = improved_rows
                best_scope = improved_scope
    
    # Phase 3: More randomized restarts if time remains
    if best_rows is not None and time.time() < deadline - 5:
        for seed in range(100, 500, 50):
            if time.time() > deadline:
                break
            rng3 = random.Random(seed)
            strategy = rng3.choice([0, 1, 2])
            rows = _construct_greedy(n_rows, k, rng3, strategy)
            if _validate(rows):
                scope = _scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    # Quick local improvement
                    if time.time() < deadline - 10:
                        used = set()
                        for row in best_rows:
                            used |= _row_diffs(row)
                        improved_rows, improved_scope = _local_improve(
                            best_rows, n_rows, k, used, rng3, deadline - 5
                        )
                        if improved_scope < best_scope:
                            best_rows = improved_rows
                            best_scope = improved_scope
    
    if best_rows is None:
        # Fallback: pure greedy (should always work)
        rng4 = random.Random(0)
        best_rows = _construct_greedy(n_rows, k, rng4, 0)
    
    return best_rows
# EVOLVE-BLOCK-END
