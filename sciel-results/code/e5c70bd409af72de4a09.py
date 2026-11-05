"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

_TIME_LIMIT = 150.0
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
              'lookahead' (score candidates by future options)
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
            for cand in range(base, base + 60):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append(cand)
                if len(valid_cands) >= 10:
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
            for cand in valid_cands[:8]:
                new = {cand - a for a in row}
                next_count = 0
                for nc in range(cand + 1, cand + 30):
                    nd = {nc - a for a in row}
                    nd.add(nc - cand)
                    if not (nd & (used | new)):
                        next_count += 1
                    if next_count >= 5:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand
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


def _simulated_annealing(rows, n_rows, k, rng, max_iters=100000):
    """
    Simulated annealing to reduce the scope.
    Moves: pick a row, pick a position, try a new value.
    Accept if valid and (improves scope or with probability exp(-delta/T)).
    """
    rows = [r[:] for r in rows]
    best_rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    current_scope = best_scope
    
    # Precompute used diffs for each row
    row_diffs = [_row_diffs(r) for r in rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    start_temp = 50.0
    end_temp = 0.1
    cooling = 0.9995
    
    temp = start_temp
    iters = 0
    no_improve = 0
    
    while iters < max_iters and _time_left() > 5.0:
        iters += 1
        
        # Pick a random row
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        
        # Pick a random position (not 0, since row[0]=0 always)
        if len(row) <= 1:
            continue
        pos = rng.randint(1, len(row) - 1)
        
        # Try a new value near the current one
        current_val = row[pos]
        # Determine valid range
        lower = row[pos - 1] + 1 if pos > 0 else 1
        upper = row[pos + 1] - 1 if pos < len(row) - 1 else current_scope + 20
        
        # Try a few random candidates
        tried = set()
        for _ in range(5):
            if pos == len(row) - 1:
                # Last element: try reducing
                if current_val > lower:
                    new_val = rng.randint(lower, current_val)
                else:
                    new_val = current_val + rng.randint(1, 10)
            else:
                # Middle element
                if upper > lower:
                    new_val = rng.randint(lower, upper)
                else:
                    continue
            
            if new_val in tried or new_val == current_val:
                continue
            tried.add(new_val)
            
            # Check internal validity
            new_row = row[:pos] + [new_val] + row[pos + 1:]
            # Check strictly increasing
            if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                continue
            
            # Check internal differences are all distinct
            internal = _row_diffs(new_row)
            expected_count = len(new_row) * (len(new_row) - 1) // 2
            if len(internal) != expected_count:
                continue
            
            # Check against other rows' differences
            other_diffs = set()
            for i, rd in enumerate(row_diffs):
                if i != r_idx:
                    other_diffs |= rd
            
            if internal & other_diffs:
                continue
            
            # Valid move! Compute new scope
            new_scope = max(max(new_row), max(max(rows[i]) for i in range(n_rows) if i != r_idx))
            
            # Acceptance criterion
            if new_scope < current_scope:
                # Always accept improvement
                rows[r_idx] = new_row
                row_diffs[r_idx] = internal
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Worse or same: accept with probability
                delta = new_scope - current_scope
                if temp > 0 and rng.random() < math.exp(-delta / temp):
                    rows[r_idx] = new_row
                    row_diffs[r_idx] = internal
                    current_scope = new_scope
                    no_improve += 1
                else:
                    no_improve += 1
            
            break
        
        # Update temperature
        temp = max(end_temp, temp * cooling)
        
        # If no improvement for a long time and temp is low, restart
        if no_improve > 5000 and temp < 1.0:
            # Restart from best
            rows = [r[:] for r in best_rows]
            row_diffs = [_row_diffs(r) for r in rows]
            current_scope = best_scope
            no_improve = 0
            temp = start_temp * 0.5
    
    return best_rows, best_scope


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
            target = best_scope - 1
        else:
            target = best_scope - 1
        
        if target < k + 1:
            break
        
        # Compute differences used by other rows
        used_other = _all_diffs_except(rows, max_idx)
        
        # Try to rebuild the max row with smaller scope using backtracking
        row = rows[max_idx]
        improved = False
        
        # Try reducing the last element
        for new_last in range(min(current_max - 1, target), row[-2], -1):
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
                upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else target
                for new_val in range(min(old_val - 1, upper_bound), lower_bound - 1, -1):
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
                            # Try to rebuild alt row
                            alt_row = rows[alt_idx]
                            for pos in range(len(alt_row) - 1, 0, -1):
                                old_val = alt_row[pos]
                                lower_bound = alt_row[pos - 1] + 1
                                upper_bound = alt_row[pos + 1] - 1 if pos < len(alt_row) - 1 else target_alt
                                for new_val in range(min(old_val - 1, upper_bound), lower_bound - 1, -1):
                                    if new_val < lower_bound:
                                        break
                                    new_alt_row = alt_row[:pos] + [new_val] + alt_row[pos + 1:]
                                    if any(new_alt_row[i] >= new_alt_row[i + 1] for i in range(len(new_alt_row) - 1)):
                                        continue
                                    internal_alt = _row_diffs(new_alt_row)
                                    if len(internal_alt) != len(new_alt_row) * (len(new_alt_row) - 1) // 2:
                                        continue
                                    if internal_alt & used_other_alt:
                                        continue
                                    rows[alt_idx] = new_alt_row
                                    new_scope = _scope(rows)
                                    if new_scope < best_scope:
                                        best_scope = new_scope
                                    found_move = True
                                    break
                                if found_move:
                                    break
                            if found_move:
                                continue
    
    return rows, best_scope


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(42)

    # Phase 1: Diverse randomized constructions
    strategies = ["pure", "shuffled", "lookahead", "shuffled", "lookahead", "pure", "shuffled"]
    best_rows = None
    best_scope = float('inf')
    
    for i, strategy in enumerate(strategies):
        if _time_left() < 20.0:
            break
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
    
    # More randomized restarts
    attempts = 0
    while _time_left() > 25.0 and attempts < 80:
        attempts += 1
        strategy = rng.choice(["pure", "shuffled", "lookahead"])
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 112:
                break

    # Phase 2: Simulated annealing
    if _time_left() > 30.0:
        rows_sa, scope_sa = _simulated_annealing(best_rows, n_rows, k, rng, max_iters=80000)
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa

    # Phase 3: Local search to reduce scope
    if _time_left() > 15.0:
        rows_ls, scope_ls = _local_search(best_rows, n_rows, k, rng, max_iters=40000)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls

    # Phase 4: Another SA pass
    if _time_left() > 15.0:
        rows_sa2, scope_sa2 = _simulated_annealing(best_rows, n_rows, k, rng, max_iters=50000)
        if scope_sa2 < best_scope:
            best_scope = scope_sa2
            best_rows = rows_sa2

    # Phase 5: Final local search
    if _time_left() > 8.0:
        rows_ls2, scope_ls2 = _local_search(best_rows, n_rows, k, rng, max_iters=20000)
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
