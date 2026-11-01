"""Difference triangle set (4,7) - randomized local search with SA."""
import time
import random
import copy

def _validate(rows):
    """Check that all positive differences across all rows are distinct."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    """Return the largest entry (scope)."""
    return max(max(r) for r in rows)

def _greedy_row(used, k, rng, max_val=300):
    """Build one row greedily with some randomness to avoid deterministic traps."""
    row = [0]
    while len(row) < k + 1:
        # Try candidates in a randomized order near the smallest feasible
        start = row[-1] + 1
        # Collect feasible candidates
        candidates = []
        for x in range(start, max_val):
            new = {x - a for a in row}
            if not new & used:
                candidates.append(x)
            if len(candidates) >= 5:
                break
        if not candidates:
            return None
        # Pick with bias toward smaller values but with randomness
        if rng.random() < 0.7:
            x = candidates[0]
        else:
            x = candidates[rng.randint(0, len(candidates) - 1)]
        row.append(x)
        used.update({x - a for a in row[:-1]})
    return row

def _greedy_construct(n_rows, k, rng, max_val=300):
    """Build all rows greedily."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_val)
        if row is None:
            return None
        rows.append(row)
    return rows

def _perturb_row(row, k, rng, used, max_val=300):
    """Perturb one row: remove a random element and re-insert, or shift."""
    new_row = list(row)
    # Pick a random position (not 0) to remove
    if len(new_row) <= 1:
        return new_row
    pos = rng.randint(1, len(new_row) - 1)
    removed = new_row.pop(pos)
    # Try to re-insert at a different position
    attempts = 0
    while attempts < 20:
        # Try a random position
        insert_pos = rng.randint(0, len(new_row))
        # Find a value that fits
        for x in range(1, max_val):
            test_row = new_row[:insert_pos] + [x] + new_row[insert_pos:]
            if test_row != sorted(test_row):
                continue
            if test_row[0] != 0:
                continue
            # Check differences
            new_diffs = set()
            valid = True
            for i in range(len(test_row)):
                for j in range(i + 1, len(test_row)):
                    d = test_row[j] - test_row[i]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            if valid:
                # Check against used (excluding this row's old diffs)
                # We'll handle conflict checking in the main loop
                return test_row
        attempts += 1
    # If we couldn't find a good re-insert, try shifting a random element
    new_row = list(row)
    pos = rng.randint(1, len(new_row) - 1)
    shift = rng.choice([-2, -1, 1, 2, 3])
    new_row[pos] += shift
    if new_row[pos] <= new_row[pos - 1] or (pos < len(new_row) - 1 and new_row[pos] >= new_row[pos + 1]):
        # Invalid ordering, revert
        return list(row)
    return new_row

def _compute_all_diffs(rows):
    """Return the set of all differences."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used

def _count_conflicts(rows):
    """Count how many duplicate differences exist."""
    from collections import Counter
    diffs = []
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
    c = Counter(diffs)
    conflicts = 0
    for d, cnt in c.items():
        if cnt > 1:
            conflicts += cnt - 1
    return conflicts

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Phase 1: Greedy construction to get a valid baseline
    best_rows = None
    best_scope = float('inf')
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        rows = _greedy_construct(n_rows, k, rng, max_val=250)
        if rows is not None and _validate(rows):
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = copy.deepcopy(rows)
    
    if best_rows is None:
        # Fallback: pure deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)
    
    if time.time() > deadline - 10:
        return best_rows
    
    # Phase 2: Local search with simulated annealing
    current_rows = copy.deepcopy(best_rows)
    current_scope = best_scope
    current_conflicts = _count_conflicts(current_rows)
    
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline - 5:
        # Pick a random row to perturb
        row_idx = rng.randint(0, n_rows - 1)
        old_row = current_rows[row_idx]
        
        # Compute diffs of other rows (fixed)
        other_diffs = set()
        for i, row in enumerate(current_rows):
            if i == row_idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    other_diffs.add(row[b] - row[a])
        
        # Try to build a new version of this row
        new_row = _greedy_row(other_diffs, k, rng, max_val=best_scope + 20)
        if new_row is None:
            # Try perturbation instead
            new_row = _perturb_row(old_row, k, rng, other_diffs, max_val=best_scope + 20)
        
        # Check if new row is valid (no internal conflicts and no conflict with others)
        new_diffs = set()
        valid = True
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                d = new_row[j] - new_row[i]
                if d in new_diffs or d in other_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                break
        
        if valid:
            new_scope = _scope([new_row] + [current_rows[i] for i in range(n_rows) if i != row_idx])
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < (10 ** (-delta / max(T, 1e-6))):
                current_rows[row_idx] = new_row
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = copy.deepcopy(current_rows)
        else:
            # Accept with probability based on conflict count
            new_conflicts = _count_conflicts([new_row] + [current_rows[i] for i in range(n_rows) if i != row_idx])
            delta_conf = new_conflicts - current_conflicts
            if delta_conf <= 0 or rng.random() < (10 ** (-delta_conf / max(T, 1e-6))):
                current_rows[row_idx] = new_row
                current_conflicts = new_conflicts
        
        T *= cooling
    
    # Phase 3: Final polish - try to reduce scope by tightening rows
    if time.time() < deadline - 3:
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            for row_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break
                other_diffs = set()
                for i, row in enumerate(best_rows):
                    if i == row_idx:
                        continue
                    for a in range(len(row)):
                        for b in range(a + 1, len(row)):
                            other_diffs.add(row[b] - row[a])
                # Try to rebuild this row with tighter max
                target_max = best_rows[row_idx][-1] - 1
                if target_max < k:
                    continue
                new_row = _greedy_row(other_diffs, k, rng, max_val=target_max)
                if new_row is not None:
                    new_diffs = set()
                    valid = True
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            if d in new_diffs or d in other_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if not valid:
                            break
                    if valid and _scope([new_row] + [best_rows[i] for i in range(n_rows) if i != row_idx]) < best_scope:
                        best_rows[row_idx] = new_row
                        best_scope = _scope(best_rows)
    
    return best_rows
