"""Difference triangle set construction using simulated annealing on permutation/assignment."""
import time
import random
import sys

_DEADLINE = 145.0
_START_TIME = None


def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0]] * n_rows
    
    # Phase 1: Greedy baseline
    baseline = _greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in baseline)
    best_rows = baseline
    
    # Phase 2: Local search with hill climbing
    rng = random.Random(12345)
    
    improved = _local_search_aggressive(best_rows, n_rows, k, rng)
    if improved is not None:
        scope = max(max(row) for row in improved)
        if scope < best_scope:
            best_scope = scope
            best_rows = improved
    
    # Phase 3: Simulated annealing with restarts
    for attempt in range(500000):
        if time.time() - _START_TIME > _DEADLINE:
            break
        
        # Random restart with annealing
        rows = _anneal(n_rows, k, rng, best_scope)
        if rows is not None:
            scope = max(max(row) for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                if best_scope <= 112:
                    break
        elif attempt % 100 == 0:
            # Try local search on best
            improved = _local_search_aggressive(best_rows, n_rows, k, rng)
            if improved is not None:
                scope = max(max(row) for row in improved)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = improved
                    if best_scope <= 112:
                        break
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Standard greedy: each row takes smallest available marks."""
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


def _local_search_aggressive(rows, n_rows, k, rng):
    """Aggressive local search: repeatedly try to reduce scope by moving elements."""
    rows = [list(r) for r in rows]
    current_scope = max(max(row) for row in rows)
    improved = True
    max_iter = 1000
    
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        
        # Compute all used differences
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        # Find the row with the maximum element
        max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        row = rows[max_row_idx]
        max_val = row[-1]
        
        if max_val > current_scope:
            current_scope = max_val
        
        # Try to reduce the last element of the max row
        last_elem_diffs = {max_val - a for a in row[:-1]}
        other_used = set()
        for i, r in enumerate(rows):
            if i == max_row_idx:
                continue
            for j in range(len(r)):
                for jj in range(j + 1, len(r)):
                    other_used.add(r[jj] - r[j])
        
        internal_used = set()
        for j in range(len(row) - 1):
            for jj in range(j + 1, len(row) - 1):
                internal_used.add(row[jj] - row[j])
        
        forbidden = other_used | internal_used
        
        # Try to find a smaller value
        for candidate in range(max_val - 1, row[-2], -1):
            cand_diffs = {candidate - a for a in row[:-1]}
            if not cand_diffs & forbidden:
                rows[max_row_idx] = row[:-1] + [candidate]
                improved = True
                break
        
        if not improved:
            # Try moving non-last elements too
            for r_idx in range(n_rows):
                r = rows[r_idx]
                if len(r) < 2:
                    continue
                # Try to reduce the last element of this row
                max_val_r = r[-1]
                last_diffs = {max_val_r - a for a in r[:-1]}
                
                # Collect forbidden differences (excluding those from this row's last element)
                forb = set()
                for i, rr in enumerate(rows):
                    if i == r_idx:
                        # Only internal diffs not involving last element
                        for j in range(len(rr) - 1):
                            for jj in range(j + 1, len(rr) - 1):
                                forb.add(rr[jj] - rr[j])
                    else:
                        for j in range(len(rr)):
                            for jj in range(j + 1, len(rr)):
                                forb.add(rr[jj] - rr[j])
                
                for candidate in range(max_val_r - 1, r[-2], -1):
                    cand_diffs = {candidate - a for a in r[:-1]}
                    if not cand_diffs & forb:
                        rows[r_idx] = r[:-1] + [candidate]
                        improved = True
                        break
                if improved:
                    break
    
    if improved:
        final_scope = max(max(row) for row in rows)
        if final_scope < current_scope:
            return rows
    return None


def _anneal(n_rows, k, rng, max_scope_hint):
    """Simulated annealing to construct a valid difference triangle set."""
    # Start from a greedy-like solution
    rows = _greedy_construct(n_rows, k)
    current_scope = max(max(row) for row in rows)
    
    if current_scope >= max_scope_hint:
        return None
    
    # Temperature schedule
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    for step in range(5000):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        if T < T_min:
            break
        
        # Random move: pick a random row and element, try to change it
        r_idx = rng.randrange(n_rows)
        row = rows[r_idx]
        pos = rng.randrange(1, len(row))  # Don't change 0
        
        old_val = row[pos]
        
        # Compute used differences
        used_diffs = set()
        for i, r in enumerate(rows):
            for j in range(len(r)):
                for jj in range(j + 1, len(r)):
                    if not (i == r_idx and (j == pos or jj == pos)):
                        used_diffs.add(r[jj] - r[j])
        
        # Try a few candidate values
        candidates = []
        for delta in range(-5, 6):
            new_val = old_val + delta
            if new_val > row[pos - 1] and (pos == len(row) - 1 or new_val < row[pos + 1]):
                if new_val >= 0:
                    candidates.append(new_val)
        
        if not candidates:
            T *= cooling
            continue
        
        new_val = rng.choice(candidates)
        if new_val == old_val:
            continue
        
        # Check validity
        row_new = list(row)
        row_new[pos] = new_val
        
        # Check if strictly increasing
        valid = True
        for j in range(len(row_new) - 1):
            if row_new[j] >= row_new[j + 1]:
                valid = False
                break
        
        if not valid:
            T *= cooling
            continue
        
        # Check differences
        new_diffs = set()
        for i in range(len(row_new)):
            for j in range(i + 1, len(row_new)):
                d = row_new[j] - row_new[i]
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
        
        if not valid:
            T *= cooling
            continue
        
        # Accept with probability
        new_scope = max(max(r) for r in rows if r is not row)
        if new_val > row[-1]:
            new_scope = max(new_scope, new_val)
        
        delta_scope = new_scope - current_scope
        
        if delta_scope < 0 or rng.random() < pow(2.718, -delta_scope / max(T, 0.01)):
            rows[r_idx] = row_new
            current_scope = new_scope
        
        T *= cooling
    
    # Validate final solution
    if _validate(rows, n_rows, k):
        return rows
    return None


def _validate(rows, n_rows, k):
    """Check that all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True
