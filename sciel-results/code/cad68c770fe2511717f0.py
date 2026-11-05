"""Constraint-satisfaction search for difference triangle sets."""
import time
import random
import numpy as np

def _valid(rows):
    """Check that all positive intra-row differences are distinct across rows."""
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
    return max(max(row) for row in rows) if rows else 0

def _greedy_row(used, k):
    """Build one row greedily given the set of already-used differences."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
    return row

def _local_search(rows, k, deadline, rng):
    """
    Joint local search over all rows.
    Moves: rebuild a random row, or swap two entries within a row (if still increasing and valid).
    Accepts improvements; occasionally accepts worse moves to escape local optima.
    """
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    # Precompute used diffs for current
    def compute_used(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    current_used = compute_used(current_rows)
    
    t = 1.0
    t_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick a move
        move_type = rng.random()
        if move_type < 0.6:
            # Rebuild a random row
            idx = rng.randrange(n_rows)
            used_without = set(current_used)
            # Remove current row's diffs
            row = current_rows[idx]
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_without.discard(row[j] - row[i])
            new_row = _greedy_row(used_without, k)
            # Add new diffs
            new_used = set(used_without)
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in new_used:
                        break
                    new_used.add(d)
            else:
                new_scope = _scope([current_rows[i] if i != idx else new_row for i in range(n_rows)])
                delta = new_scope - current_scope
                if delta <= 0 or rng.random() < np.exp(-delta / t):
                    # Remove old, add new
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            current_used.discard(row[j] - row[i])
                    current_rows[idx] = new_row
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            current_used.add(new_row[j] - new_row[i])
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
        elif move_type < 0.85:
            # Swap two entries in a random row (maintain increasing order)
            idx = rng.randrange(n_rows)
            row = current_rows[idx]
            if len(row) < 2:
                continue
            i, j = rng.sample(range(len(row)), 2)
            if i > j:
                i, j = j, i
            # Try swapping values, then re-sort
            new_row = list(row)
            new_row[i], new_row[j] = new_row[j], new_row[i]
            new_row.sort()
            if new_row == row:
                continue
            # Check validity of new row
            row_diffs = set()
            valid = True
            for a in range(len(new_row)):
                for b in range(a+1, len(new_row)):
                    d = new_row[b] - new_row[a]
                    if d in row_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            if not valid:
                continue
            # Check against other rows' diffs
            used_without = set(current_used)
            old_row = row
            for a in range(len(old_row)):
                for b in range(a+1, len(old_row)):
                    used_without.discard(old_row[b] - old_row[a])
            conflict = False
            for d in row_diffs:
                if d in used_without:
                    conflict = True
                    break
            if conflict:
                continue
            new_scope = _scope([current_rows[x] if x != idx else new_row for x in range(n_rows)])
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < np.exp(-delta / t):
                # Apply
                for a in range(len(old_row)):
                    for b in range(a+1, len(old_row)):
                        current_used.discard(old_row[b] - old_row[a])
                current_rows[idx] = new_row
                for d in row_diffs:
                    current_used.add(d)
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
        else:
            # Perturb: add a random small offset to one entry and re-sort
            idx = rng.randrange(n_rows)
            row = current_rows[idx]
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            new_val = old_val + rng.randint(-3, 3)
            if new_val <= row[pos-1] or (pos < len(row)-1 and new_val >= row[pos+1]):
                continue
            new_row = list(row)
            new_row[pos] = new_val
            new_row.sort()
            if new_row == row:
                continue
            # Check validity
            row_diffs = set()
            valid = True
            for a in range(len(new_row)):
                for b in range(a+1, len(new_row)):
                    d = new_row[b] - new_row[a]
                    if d in row_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            if not valid:
                continue
            used_without = set(current_used)
            old_row = row
            for a in range(len(old_row)):
                for b in range(a+1, len(old_row)):
                    used_without.discard(old_row[b] - old_row[a])
            conflict = False
            for d in row_diffs:
                if d in used_without:
                    conflict = True
                    break
            if conflict:
                continue
            new_scope = _scope([current_rows[x] if x != idx else new_row for x in range(n_rows)])
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < np.exp(-delta / t):
                for a in range(len(old_row)):
                    for b in range(a+1, len(old_row)):
                        current_used.discard(old_row[b] - old_row[a])
                current_rows[idx] = new_row
                for d in row_diffs:
                    current_used.add(d)
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
        
        t = max(t_min, t * cooling)
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    rng = random.Random(42)
    
    # Phase 1: Greedy initial solution
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Phase 2: Local search with multiple restarts
    while time.time() < deadline:
        # Run local search
        candidate = _local_search(best_rows, k, min(time.time() + 20, deadline), rng)
        scope = _scope(candidate)
        if scope < best_scope:
            best_scope = scope
            best_rows = candidate
        
        # Occasionally restart from a fresh greedy solution
        if rng.random() < 0.3:
            used = set()
            fresh_rows = []
            for _ in range(n_rows):
                row = _greedy_row(used, k)
                fresh_rows.append(row)
            fresh_scope = _scope(fresh_rows)
            if fresh_scope < best_scope:
                best_scope = fresh_scope
                best_rows = [list(r) for r in fresh_rows]
    
    # Final validation
    if _valid(best_rows):
        return best_rows
    
    # Fallback: return greedy solution (should always be valid)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows
