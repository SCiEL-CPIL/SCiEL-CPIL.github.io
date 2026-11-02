import time
import random
import copy
import math

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 integers
    starting at 0, strictly increasing, with all positive differences distinct
    across all rows. Minimize the largest entry.
    """
    deadline = time.time() + 112.0
    rng = random.Random(42)
    
    # Compute a lower bound on M
    total_diffs_needed = n_rows * k * (k + 1) // 2
    M_lower = int(math.ceil((math.sqrt(1 + 8 * total_diffs_needed) - 1) / 2))
    
    best_rows = None
    best_score = None
    
    # Multi-start local search with row-rebuild and single-position moves
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Build initial solution
        rows = _build_initial(n_rows, k, rng, M_lower + attempt // 10)
        if rows is None:
            continue
        
        score = _max_entry(rows)
        if best_score is not None and score >= best_score:
            continue
        
        # Local search
        rows, score = _local_search(rows, n_rows, k, rng, deadline - 5.0)
        
        if best_score is None or score < best_score:
            best_score = score
            best_rows = copy.deepcopy(rows)
            if best_score == M_lower:
                break
    
    if best_rows is None:
        # Fallback: pure greedy
        best_rows = _greedy_construct(n_rows, k)
    
    return best_rows


def _max_entry(rows):
    return max(max(r) for r in rows)


def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    dset = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            dset.add(row[j] - row[i])
    return dset


def _compute_all_diffs(rows):
    """Compute the union of all differences across all rows."""
    all_diffs = set()
    for row in rows:
        all_diffs |= _row_diffs(row)
    return all_diffs


def _greedy_construct(n_rows, k):
    """
    Greedy construction: for each row, pick the smallest possible next value
    that doesn't conflict with existing differences.
    """
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


def _build_initial(n_rows, k, rng, M_target):
    """
    Build an initial solution using randomized greedy.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            base = row[-1]
            # Try values from base+1 up to M_target
            candidates = []
            for x in range(base + 1, M_target + 1):
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    candidates.append(x)
            
            if not candidates:
                # Try a few more values beyond M_target
                for x in range(base + 1, M_target + 50):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        candidates.append(x)
                        break
                if not candidates:
                    return None
            
            # Weighted random pick, biased to smaller values
            if len(candidates) > 1:
                weights = [1.0 / (c - base + 1) ** 0.8 for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                pick = candidates[-1]
                for i, c in enumerate(candidates):
                    cumulative += weights[i]
                    if r <= cumulative:
                        pick = c
                        break
            else:
                pick = candidates[0]
            
            row.append(pick)
            used |= {pick - a for a in row[:-1]}
        
        rows.append(row)
    
    return rows


def _local_search(rows, n_rows, k, rng, deadline):
    """
    Local search with two move types:
    1. Single-position move: change one element in a row
    2. Row-rebuild move: rebuild an entire row from scratch
    
    Maintains incremental difference sets for speed.
    """
    current = copy.deepcopy(rows)
    current_score = _max_entry(current)
    best = copy.deepcopy(current)
    best_score = current_score
    
    # Precompute per-row diffs and global used diffs
    row_diffs = [_row_diffs(row) for row in current]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    T = max(1.0, current_score * 0.05)
    cooling = 0.9995
    min_T = 0.01
    no_improve_steps = 0
    max_no_improve = 500
    
    while time.time() < deadline and T > min_T and no_improve_steps < max_no_improve:
        # Choose move type: 70% single-position, 30% row-rebuild
        if rng.random() < 0.7:
            move_type = 'single'
        else:
            move_type = 'rebuild'
        
        new_current = None
        new_score = None
        new_row_diffs = None
        new_all_diffs = None
        
        if move_type == 'single':
            result = _try_single_pos_move(current, row_diffs, all_diffs, 
                                          n_rows, k, rng, T)
            if result is not None:
                new_row_idx, new_row, new_score = result
                new_current = copy.deepcopy(current)
                new_current[new_row_idx] = new_row
                new_row_diffs = copy.deepcopy(row_diffs)
                new_row_diffs[new_row_idx] = _row_diffs(new_row)
                new_all_diffs = set()
                for rd in new_row_diffs:
                    new_all_diffs |= rd
        else:
            result = _try_row_rebuild(current, row_diffs, all_diffs,
                                      n_rows, k, rng, T)
            if result is not None:
                new_row_idx, new_row, new_score = result
                new_current = copy.deepcopy(current)
                new_current[new_row_idx] = new_row
                new_row_diffs = copy.deepcopy(row_diffs)
                new_row_diffs[new_row_idx] = _row_diffs(new_row)
                new_all_diffs = set()
                for rd in new_row_diffs:
                    new_all_diffs |= rd
        
        if new_current is None:
            T *= cooling
            continue
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < _accept_probability(delta, T):
            # Accept move
            current = new_current
            current_score = new_score
            row_diffs = new_row_diffs
            all_diffs = new_all_diffs
            
            if new_score < best_score:
                best_score = new_score
                best = copy.deepcopy(current)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
        
        T *= cooling
    
    return best, best_score


def _try_single_pos_move(current, row_diffs, all_diffs, n_rows, k, rng, T):
    """
    Try a single-position move: change one element in a random row.
    Returns (row_idx, new_row, new_score) or None.
    """
    row_idx = rng.randint(0, n_rows - 1)
    pos = rng.randint(1, k)
    row = current[row_idx]
    old_val = row[pos]
    
    lo = row[pos - 1] + 1
    hi = row[pos + 1] - 1 if pos + 1 < k + 1 else _max_entry(current) + 10
    
    if hi < lo:
        return None
    
    # Generate candidates
    candidates = set()
    # Near old value
    for d in range(-3, 4):
        v = old_val + d
        if lo <= v <= hi and v != old_val:
            candidates.add(v)
    # Random values
    for _ in range(5):
        v = rng.randint(lo, hi)
        if v != old_val:
            candidates.add(v)
    # Boundary values
    candidates.add(lo)
    candidates.add(hi)
    
    # Also try to minimize: try lo specifically
    for new_val in candidates:
        if new_val == old_val:
            continue
        if new_val <= row[pos - 1] or (pos + 1 < k + 1 and new_val >= row[pos + 1]):
            continue
        
        new_row = row[:]
        new_row[pos] = new_val
        new_row_diff = _row_diffs(new_row)
        
        # Check conflicts with other rows
        other_diffs = set()
        for i in range(n_rows):
            if i != row_idx:
                other_diffs |= row_diffs[i]
        
        if new_row_diff & other_diffs:
            continue
        
        # Valid move
        new_score = max(_max_entry(current[:row_idx] + [new_row] + current[row_idx+1:]), new_row[-1])
        return row_idx, new_row, new_score
    
    return None


def _try_row_rebuild(current, row_diffs, all_diffs, n_rows, k, rng, T):
    """
    Try to rebuild a random row from scratch.
    Returns (row_idx, new_row, new_score) or None.
    """
    row_idx = rng.randint(0, n_rows - 1)
    
    # Diffs from other rows
    other_diffs = set()
    for i in range(n_rows):
        if i != row_idx:
            other_diffs |= row_diffs[i]
    
    # Build new row avoiding other_diffs
    new_row = [0]
    used = set(other_diffs)
    max_target = _max_entry(current) + 5
    
    while len(new_row) < k + 1:
        base = new_row[-1]
        candidates = []
        for x in range(base + 1, max_target + 1):
            new_diffs = {x - a for a in new_row}
            if len(new_diffs) == len(new_row) and not (new_diffs & used):
                candidates.append(x)
        
        if not candidates:
            return None
        
        # Weighted pick
        if len(candidates) > 1:
            weights = [1.0 / (c - base + 1) ** 0.8 for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            pick = candidates[-1]
            for i, c in enumerate(candidates):
                cumulative += weights[i]
                if r <= cumulative:
                    pick = c
                    break
        else:
            pick = candidates[0]
        
        new_row.append(pick)
        used |= {pick - a for a in new_row[:-1]}
    
    new_score = max(_max_entry(current[:row_idx] + [new_row] + current[row_idx+1:]), new_row[-1])
    return row_idx, new_row, new_score


def _accept_probability(delta, T):
    """Metropolis acceptance probability."""
    if delta <= 0:
        return 1.0
    if T <= 0:
        return 0.0
    val = -delta / T
    if val < -50:
        return 0.0
    return math.exp(val)
