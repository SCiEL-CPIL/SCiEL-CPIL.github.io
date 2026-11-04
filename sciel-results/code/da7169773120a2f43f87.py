"""Difference triangle set (7,5) - simulated annealing with coordinated row moves."""
import time
import random

_DEADLINE = 120.0  # safety margin under the 150s wall-clock limit


def _row_diffs(row):
    """All positive differences a_j - a_i for i < j in a single row."""
    d = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ai)
    return d


def _all_diffs(rows):
    """All differences across all rows."""
    diffs = []
    for r in rows:
        diffs.extend(_row_diffs(r))
    return diffs


def _scope(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences are distinct."""
    all_d = _all_diffs(rows)
    return len(all_d) == len(set(all_d))


def _greedy(n_rows, k, used=None, rows=None, start_offset=0):
    """Greedy ruler construction. Returns (rows, used)."""
    if used is None:
        used = set()
    if rows is None:
        rows = []
    for r_idx in range(n_rows):
        row, x = [0], start_offset
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows, used


def _sa_step(rows, k, used_set, temperature, rng):
    """
    Perform one simulated annealing step.
    Returns (new_rows, new_used_set, accepted).
    """
    n_rows = len(rows)
    
    # Choose a random row to modify
    row_idx = rng.randint(0, n_rows - 1)
    row = rows[row_idx]
    
    # Choose a random position (not 0) to modify
    pos = rng.randint(1, len(row) - 1)
    
    # Determine valid range for new value
    earlier = row[:pos]
    later = row[pos+1:]
    
    low = (earlier[-1] + 1) if earlier else 1
    high = (later[0] - 1) if later else row[pos] + 5  # allow some expansion
    
    if low > high:
        return rows, used_set, False
    
    # Generate a candidate new value (biased toward smaller values)
    old_val = row[pos]
    
    # Try to move the element: pick a new value from a range
    # Bias toward smaller values to reduce scope
    max_search = min(30, high - low + 1)
    candidates = []
    
    # Always include values below current (to reduce scope)
    for v in range(max(low, old_val - 20), min(old_val, high + 1)):
        if v != old_val:
            candidates.append(v)
    
    # Also include some values above (to escape local minima)
    for v in range(old_val + 1, min(old_val + 10, high + 1)):
        candidates.append(v)
    
    if not candidates:
        return rows, used_set, False
    
    new_val = rng.choice(candidates)
    if new_val == old_val:
        return rows, used_set, False
    
    # Compute old differences involving this element
    old_diffs = set()
    for a in earlier:
        old_diffs.add(old_val - a)
    for b in later:
        old_diffs.add(b - old_val)
    
    # Compute new differences
    new_diffs = set()
    for a in earlier:
        new_diffs.add(new_val - a)
    for b in later:
        new_diffs.add(b - new_val)
    
    # Check if new differences conflict with used set (excluding old_diffs)
    used_without_old = used_set - old_diffs
    
    # Count conflicts
    conflicts = len(new_diffs & used_without_old)
    
    # Also check for internal conflicts in new_diffs (shouldn't happen if row is strictly increasing)
    # But let's be safe
    if len(new_diffs) != len(new_diffs):  # This check is redundant but safe
        return rows, used_set, False
    
    if conflicts == 0:
        # Accept: no conflicts
        new_row = row.copy()
        new_row[pos] = new_val
        new_rows = rows.copy()
        new_rows[row_idx] = new_row
        new_used = used_without_old | new_diffs
        return new_rows, new_used, True
    else:
        # Conflict: accept with probability exp(-conflicts / temperature)
        if rng.random() < (2.718281828 ** (-conflicts / max(temperature, 1e-10))):
            new_row = row.copy()
            new_row[pos] = new_val
            new_rows = rows.copy()
            new_rows[row_idx] = new_row
            # Build new used set: remove old_diffs, add new_diffs
            # But we need to handle the fact that new_diffs may have conflicts
            # For SA, we allow temporary invalid states
            new_used = used_without_old | new_diffs
            return new_rows, new_used, True
        else:
            return rows, used_set, False


def _simulated_annealing(rows, k, deadline, rng_seed=42):
    """
    Run simulated annealing on the rows.
    Returns the best valid solution found.
    """
    rng = random.Random(rng_seed)
    start_time = time.time()
    
    # Initial state
    current_rows = [list(r) for r in rows]
    current_used = set()
    for r in current_rows:
        current_used.update(_row_diffs(r))
    
    best_rows = [list(r) for r in current_rows]
    best_scope = _scope(best_rows)
    
    # Check if initial is valid
    if _is_valid(current_rows):
        best_rows = [list(r) for r in current_rows]
        best_scope = _scope(best_rows)
    
    # Annealing schedule
    t0 = time.time()
    initial_temp = 5.0
    final_temp = 0.01
    total_time = deadline - t0
    
    # Number of steps we want to try (will be limited by time)
    step_count = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 2.0:
        # Linear cooling
        elapsed = time.time() - t0
        progress = min(1.0, elapsed / max(total_time, 1.0))
        temperature = initial_temp * (1 - progress) + final_temp * progress
        
        # Perform a batch of steps
        batch_size = 100
        for _ in range(batch_size):
            if time.time() > deadline - 2.0:
                break
            
            new_rows, new_used, accepted = _sa_step(current_rows, k, current_used, temperature, rng)
            
            if accepted:
                current_rows = new_rows
                current_used = new_used
                step_count += 1
                
                # Check if current state is valid
                if _is_valid(current_rows):
                    scope = _scope(current_rows)
                    if scope < best_scope:
                        best_rows = [list(r) for r in current_rows]
                        best_scope = scope
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        
        # Occasionally restart from best if stuck
        if no_improve_count > max_no_improve:
            if rng.random() < 0.3:
                current_rows = [list(r) for r in best_rows]
                current_used = set()
                for r in current_rows:
                    current_used.update(_row_diffs(r))
            no_improve_count = 0
    
    return best_rows


def _local_search(rows, used, k, deadline):
    """Repeatedly try to shrink rows by moving any element. Returns (rows, used)."""
    while time.time() < deadline:
        any_improved = False
        for i in range(len(rows)):
            row = rows[i]
            for p in range(len(row) - 1, 0, -1):
                # Try to reduce element at position p
                old_val = row[p]
                earlier = row[:p]
                later = row[p+1:]
                
                old_diffs = set()
                for a in earlier:
                    old_diffs.add(old_val - a)
                for b in later:
                    old_diffs.add(b - old_val)
                
                low = (earlier[-1] + 1) if earlier else 1
                high = (later[0] - 1) if later else old_val
                
                if low > high:
                    continue
                
                for v in range(old_val - 1, low - 1, -1):
                    cand_diffs = set()
                    for a in earlier:
                        cand_diffs.add(v - a)
                    for b in later:
                        cand_diffs.add(b - v)
                    
                    if not (cand_diffs & (used - old_diffs)):
                        used -= old_diffs
                        used |= cand_diffs
                        row[p] = v
                        any_improved = True
                        break
            if time.time() > deadline:
                break
        if not any_improved:
            break
    return rows, used


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE - 5.0  # safety margin

    # Phase 1: fast greedy baseline (always valid)
    rows, used = _greedy(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)

    # Phase 2: local search to shrink
    rows_ls, used_ls = _local_search([list(r) for r in rows], set(used), k, min(deadline, start + 10.0))
    if _is_valid(rows_ls) and _scope(rows_ls) < best_scope:
        best_rows = [list(r) for r in rows_ls]
        best_scope = _scope(best_rows)

    # Phase 3: Simulated annealing with multiple restarts
    rng_seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    for seed in rng_seeds:
        if time.time() > deadline - 15.0:
            break
        
        # Start from the best found so far
        sa_rows = _simulated_annealing(
            [list(r) for r in best_rows], 
            k, 
            min(deadline - 5.0, time.time() + 10.0),
            rng_seed=seed
        )
        
        if _is_valid(sa_rows):
            scope_sa = _scope(sa_rows)
            if scope_sa < best_scope:
                best_rows = [list(r) for r in sa_rows]
                best_scope = scope_sa
                
                # If improved, do a quick local search
                if time.time() < deadline - 10.0:
                    used_tmp = set()
                    for r in best_rows:
                        used_tmp.update(_row_diffs(r))
                    ls_rows, _ = _local_search([list(r) for r in best_rows], used_tmp, k, min(deadline - 5.0, time.time() + 3.0))
                    if _is_valid(ls_rows) and _scope(ls_rows) < best_scope:
                        best_rows = [list(r) for r in ls_rows]
                        best_scope = _scope(best_rows)
        
        # Also try SA from a fresh greedy with random offset
        if time.time() < deadline - 15.0:
            offset = random.Random(seed).randint(0, 20)
            greedy_rows, greedy_used = _greedy(n_rows, k, used=set(), rows=[], start_offset=offset)
            sa_rows2 = _simulated_annealing(
                [list(r) for r in greedy_rows],
                k,
                min(deadline - 5.0, time.time() + 8.0),
                rng_seed=seed + 10000
            )
            if _is_valid(sa_rows2):
                scope2 = _scope(sa_rows2)
                if scope2 < best_scope:
                    best_rows = [list(r) for r in sa_rows2]
                    best_scope = scope2
                    if time.time() < deadline - 10.0:
                        used_tmp = set()
                        for r in best_rows:
                            used_tmp.update(_row_diffs(r))
                        ls_rows, _ = _local_search([list(r) for r in best_rows], used_tmp, k, min(deadline - 5.0, time.time() + 3.0))
                        if _is_valid(ls_rows) and _scope(ls_rows) < best_scope:
                            best_rows = [list(r) for r in ls_rows]
                            best_scope = _scope(best_rows)

    return best_rows
