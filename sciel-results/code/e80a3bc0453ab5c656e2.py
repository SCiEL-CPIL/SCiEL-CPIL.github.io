"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math
from collections import defaultdict

_TIME_LIMIT = 120.0
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _validate(rows):
    """Check that all positive differences within each row are distinct across all rows."""
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
    return max(max(r) for r in rows) if rows else 0


def _row_diffs(row):
    """Compute set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _all_diffs_except(rows, skip_idx):
    """Compute set of all differences except those in rows[skip_idx]."""
    s = set()
    for idx, row in enumerate(rows):
        if idx == skip_idx:
            continue
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
    return s


def _greedy_row(used, k, rng=None, strategy="pure"):
    """
    Build one row greedily.
    strategy: 'pure' (smallest valid), 'shuffled' (random from window),
              'lookahead' (score candidates by future options),
              'pressure' (minimize max new difference)
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        if strategy == "pure":
            x += 1
            while True:
                new = {x - a for a in row}
                if not new & used:
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        elif strategy == "shuffled":
            base = x + 1
            window = 50
            candidates = list(range(base, base + window))
            if rng:
                rng.shuffle(candidates)
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= {cand - a for a in row[:-1]}
                    x = cand
                    found = True
                    break
            if not found:
                cand = base + window
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= {cand - a for a in row[:-1]}
                        x = cand
                        break
                    cand += 1
        elif strategy == "lookahead":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 50):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append(cand)
                if len(valid_cands) >= 8:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append(cand)
                        break
                    cand += 1
            
            best_cand = valid_cands[0]
            best_score = -1
            for cand in valid_cands[:6]:
                new = {cand - a for a in row}
                next_count = 0
                for nc in range(cand + 1, cand + 25):
                    nd = {nc - a for a in row}
                    nd.add(nc - cand)
                    if not (nd & (used | new)):
                        next_count += 1
                    if next_count >= 4:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
        elif strategy == "pressure":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 60):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append((cand, max(new)))
                if len(valid_cands) >= 10:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append((cand, max(new)))
                        break
                    cand += 1
            
            # Sort by (max_new_diff, cand) to minimize pressure
            valid_cands.sort(key=lambda t: (t[1], t[0]))
            best_cand = valid_cands[0][0]
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
    return row


def _construct_diverse(n_rows, k, rng, strategy):
    """Construct full set using a given strategy."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, strategy)
        rows.append(row)
    return rows


def _backtrack_row(used, k, max_scope, rng):
    """
    Try to build one row with all entries <= max_scope using backtracking.
    Returns row or None.
    """
    row = [0]
    
    def bt():
        if len(row) == k + 1:
            return True
        start = row[-1] + 1
        candidates = list(range(start, min(max_scope + 1, start + 60)))
        if rng and len(candidates) > 10:
            # Shuffle middle portion for diversity
            rng.shuffle(candidates[5:])
        for x in candidates:
            if x > max_scope:
                break
            new = {x - a for a in row}
            if new & used:
                continue
            row.append(x)
            used.update(new)
            if bt():
                return True
            row.pop()
            used.difference_update(new)
        return False
    
    if bt():
        return row
    return None


def _rebuild_row(used_other, k, max_scope, rng):
    """
    Rebuild a single row given the differences used by other rows.
    Returns a valid row or None.
    """
    return _backtrack_row(used_other, k, max_scope, rng)


def _local_search(rows, n_rows, k, rng, max_iters=100000):
    """
    Local search: try to reduce the largest entry by rebuilding rows.
    """
    best_scope = _scope(rows)
    rows = [r[:] for r in rows]
    iters = 0
    
    while iters < max_iters and _time_left() > 5.0:
        iters += 1
        
        # Find the row with the maximum entry
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        if current_max <= best_scope:
            # Try to reduce further
            target = best_scope - 1
        else:
            target = best_scope - 1
        
        if target < k + 1:  # Can't go below k+1
            break
        
        # Compute differences used by other rows
        used_other = _all_diffs_except(rows, max_idx)
        
        # Try to rebuild the max row with smaller scope
        new_row = _backtrack_row(used_other, k, target, rng)
        if new_row is not None:
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            continue
        
        # If rebuild fails, try reducing individual elements
        row = rows[max_idx]
        improved = False
        
        # Try reducing the last element
        for new_last in range(current_max - 1, row[-2], -1):
            if new_last <= row[-2]:
                break
            new_row = row[:-1] + [new_last]
            internal = _row_diffs(new_row)
            if len(internal) != len(new_row) * (len(new_row) - 1) // 2:
                continue
            if internal & used_other:
                continue
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            improved = True
            break
        
        if not improved:
            # Try reducing other elements
            found_move = False
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val < lower_bound:
                        break
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                        continue
                    internal = _row_diffs(new_row)
                    if len(internal) != len(new_row) * (len(new_row) - 1) // 2:
                        continue
                    if internal & used_other:
                        continue
                    rows[max_idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    found_move = True
                    break
                if found_move:
                    break
            
            if not found_move:
                # Try a different row
                if rng.random() < 0.5:
                    alt_idx = rng.randint(0, n_rows - 1)
                    if alt_idx != max_idx:
                        used_other_alt = _all_diffs_except(rows, alt_idx)
                        target_alt = min(rows[alt_idx][-1] - 1, best_scope - 1)
                        if target_alt >= k + 1:
                            new_row_alt = _backtrack_row(used_other_alt, k, target_alt, rng)
                            if new_row_alt is not None:
                                rows[alt_idx] = new_row_alt
                                new_scope = _scope(rows)
                                if new_scope < best_scope:
                                    best_scope = new_scope

    return rows, best_scope


def _iterative_tightening(rows, n_rows, k, rng):
    """
    Iteratively try to find solutions with smaller and smaller max scope.
    Uses backtracking to rebuild rows under a scope constraint.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    
    # Try decreasing scope targets
    for target in range(best_scope - 1, k + 1, -1):
        if _time_left() < 10.0:
            break
        
        # Try to construct all rows with scope <= target
        success = True
        new_rows = []
        used = set()
        
        for r_idx in range(n_rows):
            if _time_left() < 5.0:
                success = False
                break
            row = _backtrack_row(used, k, target, rng)
            if row is None:
                # Try with a slightly larger scope for this row
                row = _backtrack_row(used, k, target + 2, rng)
                if row is None:
                    success = False
                    break
            new_rows.append(row)
            used |= _row_diffs(row)
        
        if success:
            scope = _scope(new_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = new_rows
            # Continue trying smaller targets
    
    return best_rows, best_scope


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(42)

    # Phase 1: Diverse randomized constructions with pressure strategy
    strategies = ["pure", "shuffled", "lookahead", "pressure", "shuffled", "lookahead", "pure", "pressure"]
    best_rows = None
    best_scope = float('inf')
    
    for i, strategy in enumerate(strategies):
        if _time_left() < 15.0:
            break
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
    
    # Also try more randomized restarts with pressure
    attempts = 0
    while _time_left() > 20.0 and attempts < 80:
        attempts += 1
        strategy = rng.choice(["pure", "shuffled", "lookahead", "pressure"])
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 112:
                break

    # Phase 2: Local search to reduce scope
    if _time_left() > 15.0:
        rows_ls, scope_ls = _local_search(best_rows, n_rows, k, rng, max_iters=60000)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls

    # Phase 3: Iterative tightening with backtracking
    if _time_left() > 15.0:
        rows_it, scope_it = _iterative_tightening(best_rows, n_rows, k, rng)
        if scope_it < best_scope:
            best_scope = scope_it
            best_rows = rows_it

    # Phase 4: Additional pressure-based restarts if time permits
    if _time_left() > 20.0:
        extra_attempts = 0
        while _time_left() > 12.0 and extra_attempts < 30:
            extra_attempts += 1
            strategy = rng.choice(["pressure", "shuffled", "lookahead"])
            rows = _construct_diverse(n_rows, k, rng, strategy)
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= 112:
                    break

    # Phase 5: Final local search pass
    if _time_left() > 8.0:
        rows_ls2, scope_ls2 = _local_search(best_rows, n_rows, k, rng, max_iters=40000)
        if scope_ls2 < best_scope:
            best_scope = scope_ls2
            best_rows = rows_ls2

    # Final validation
    if not _validate(best_rows):
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, None, "pure")
            rows.append(row)
        return rows

    return best_rows
