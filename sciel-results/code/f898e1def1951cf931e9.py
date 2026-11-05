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
            window = 40
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
    return row


def _construct_diverse(n_rows, k, rng, strategy):
    """Construct full set using a given strategy."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, strategy)
        rows.append(row)
    return rows


def _simulated_annealing(rows, n_rows, k, rng, time_budget=60.0):
    """
    Simulated annealing on the difference coloring problem.
    State: the rows themselves.
    Energy: number of duplicate differences + penalty for large scope.
    Moves: pick a random row, change one of its entries (keep increasing order).
    """
    best_rows = [r[:] for r in rows]
    best_energy = float('inf')
    best_scope = _scope(best_rows)
    
    def compute_energy(rows):
        """Energy = #duplicate_diffs * 1000 + max_scope"""
        used = set()
        dup_count = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        dup_count += 1
                    else:
                        used.add(d)
        max_s = _scope(rows)
        return dup_count * 1000 + max_s
    
    current_rows = [r[:] for r in rows]
    current_energy = compute_energy(current_rows)
    
    T = 500.0  # initial temperature
    cooling = 0.9995
    min_T = 0.01
    
    start_time = time.time()
    while time.time() - start_time < time_budget and T > min_T:
        if _time_left() < 5.0:
            break
        
        # Pick a random row and position to modify
        row_idx = rng.randint(0, n_rows - 1)
        row = current_rows[row_idx]
        pos = rng.randint(1, len(row) - 1)  # don't change first element (must be 0)
        
        old_val = row[pos]
        
        # Determine valid range for new value
        lower = row[pos - 1] + 1
        upper = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 20
        
        if upper <= lower:
            T *= cooling
            continue
        
        # Propose a new value
        if rng.random() < 0.7:
            # Small move
            new_val = old_val + rng.choice([-2, -1, 1, 2])
        else:
            # Larger move
            new_val = rng.randint(lower, min(upper, old_val + 15))
        
        if new_val < lower or new_val > upper:
            T *= cooling
            continue
        
        # Apply move
        row[pos] = new_val
        
        # Check increasing order
        valid_order = all(row[i] < row[i + 1] for i in range(len(row) - 1))
        
        if valid_order:
            new_energy = compute_energy(current_rows)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_energy = new_energy
                # Update best
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_rows = [r[:] for r in current_rows]
                    best_scope = _scope(best_rows)
            else:
                # Reject
                row[pos] = old_val
        else:
            # Reject invalid order
            row[pos] = old_val
        
        T *= cooling
    
    return best_rows, best_scope


def _multi_restart_sa(rows, n_rows, k, rng, total_budget=90.0):
    """
    Multiple SA restarts with different initial perturbations.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    
    start_time = time.time()
    restart = 0
    while time.time() - start_time < total_budget and _time_left() > 10.0:
        restart += 1
        
        # Create perturbed initial state
        perturbed = [r[:] for r in rows]
        for _ in range(restart % 5 + 1):
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, len(perturbed[r_idx]) - 1)
            lower = perturbed[r_idx][pos - 1] + 1
            upper = perturbed[r_idx][pos + 1] - 1 if pos < len(perturbed[r_idx]) - 1 else perturbed[r_idx][pos] + 10
            if upper > lower:
                perturbed[r_idx][pos] = rng.randint(lower, upper)
        
        # Ensure valid order, fix if needed
        for r in perturbed:
            for i in range(len(r) - 1):
                if r[i] >= r[i + 1]:
                    r[i + 1] = r[i] + 1
        
        # Run SA on perturbed state
        rows_sa, scope_sa = _simulated_annealing(perturbed, n_rows, k, rng, time_budget=8.0)
        
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa
    
    return best_rows, best_scope


def _local_search(rows, n_rows, k, rng, max_iters=50000):
    """
    Local search: try to reduce the largest entry by direct manipulation.
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
        
        # Try reducing individual elements of the max row
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
                upper_bound = row[pos + 1] - 1
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val < lower_bound or new_val >= upper_bound:
                        continue
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
                # Try a random row
                if rng.random() < 0.5:
                    alt_idx = rng.randint(0, n_rows - 1)
                    if alt_idx != max_idx:
                        used_other_alt = _all_diffs_except(rows, alt_idx)
                        target_alt = min(rows[alt_idx][-1] - 1, best_scope - 1)
                        if target_alt >= k + 1:
                            row_alt = rows[alt_idx]
                            # Try reducing last element
                            for new_last in range(row_alt[-1] - 1, row_alt[-2], -1):
                                if new_last <= row_alt[-2]:
                                    break
                                new_row_alt = row_alt[:-1] + [new_last]
                                internal_alt = _row_diffs(new_row_alt)
                                if len(internal_alt) != len(new_row_alt) * (len(new_row_alt) - 1) // 2:
                                    continue
                                if internal_alt & used_other_alt:
                                    continue
                                rows[alt_idx] = new_row_alt
                                new_scope = _scope(rows)
                                if new_scope < best_scope:
                                    best_scope = new_scope
                                break

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
    while _time_left() > 30.0 and attempts < 40:
        attempts += 1
        strategy = rng.choice(["pure", "shuffled", "lookahead"])
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 112:
                break

    # Phase 2: Simulated Annealing (main optimization)
    if _time_left() > 40.0:
        rows_sa, scope_sa = _multi_restart_sa(best_rows, n_rows, k, rng, total_budget=min(50.0, _time_left() - 25.0))
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa

    # Phase 3: Local search polish
    if _time_left() > 15.0:
        rows_ls, scope_ls = _local_search(best_rows, n_rows, k, rng, max_iters=30000)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls

    # Phase 4: Final SA pass if time allows
    if _time_left() > 15.0:
        rows_sa2, scope_sa2 = _simulated_annealing(best_rows, n_rows, k, rng, time_budget=min(10.0, _time_left() - 8.0))
        if scope_sa2 < best_scope:
            best_scope = scope_sa2
            best_rows = rows_sa2

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
