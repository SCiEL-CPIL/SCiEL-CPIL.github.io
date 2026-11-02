"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Counting bound: smallest m such that m*(m+1)/2 >= total_diffs_needed
    m_lower = 0
    while m_lower * (m_lower + 1) // 2 < total_diffs_needed:
        m_lower += 1

    best_rows = None
    best_scope = float('inf')

    # First, get a quick greedy solution to establish an upper bound
    greedy_rows, greedy_scope = _greedy_construct(n_rows, k)
    best_rows = greedy_rows
    best_scope = greedy_scope

    # Try to improve with various strategies
    rng = random.Random(42)

    # Strategy 1: Randomized greedy with multiple restarts
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        rows, scope = _randomized_greedy(n_rows, k, best_scope - 1, rng, attempt)
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = rows
            # Try to compact further
            compacted, cscope = _compact_rows(rows, k, best_scope - 1, rng)
            if compacted is not None and cscope < best_scope:
                best_scope = cscope
                best_rows = compacted

    # Strategy 2: Try to find solutions at specific target scopes
    for target in range(max(m_lower, 1), best_scope):
        if time.time() > deadline - 5.0:
            break
        found = False
        for attempt in range(30):
            if time.time() > deadline - 5.0:
                break
            rows, scope = _targeted_construct(n_rows, k, target, rng, attempt)
            if rows is not None:
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                found = True
                break
        if found:
            # Found at this target, no need to try higher targets
            # But we should try to go even lower
            continue

    # Final compaction attempt
    if best_rows is not None:
        compacted, cscope = _compact_rows(best_rows, k, best_scope - 1, rng)
        if compacted is not None and cscope < best_scope:
            best_scope = cscope
            best_rows = compacted

    return best_rows


def _greedy_construct(n_rows, k):
    """Greedy: each row takes smallest next mark with unused differences."""
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
    scope = max(max(r) for r in rows)
    return rows, scope


def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _randomized_greedy(n_rows, k, max_scope, rng, seed_offset):
    """Randomized greedy construction with row shuffling."""
    rng_local = random.Random(seed_offset * 7919 + 13)
    
    row_order = list(range(n_rows))
    rng_local.shuffle(row_order)
    
    used = set()
    rows = [None] * n_rows
    
    for row_idx in row_order:
        row = _build_row_randomized(k, max_scope, used, rng_local)
        if row is None:
            return None, None
        rows[row_idx] = row
        used |= _row_diffs(row)
    
    scope = max(max(r) for r in rows)
    if scope <= max_scope:
        return rows, scope
    return None, None


def _build_row_randomized(k, max_val, used, rng):
    """Build a single row with randomization."""
    row = [0]
    current_max = 0
    
    for _ in range(k):
        candidates = []
        for x in range(current_max + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                candidates.append(x)
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            choice = candidates[0]
        else:
            # Biased toward smaller values
            temp = 0.05 + 0.2 * rng.random()
            weights = [math.exp(-c * temp / max(1, max_val)) for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            choice = candidates[-1]
            for c, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    choice = c
                    break
        
        row.append(choice)
        current_max = choice
        used |= {choice - a for a in row[:-1]}
    
    return row


def _targeted_construct(n_rows, k, target_scope, rng, seed_offset):
    """Try to construct rows with max entry <= target_scope."""
    rng_local = random.Random(seed_offset * 104729 + 31)
    
    # Try different row construction orders
    row_order = list(range(n_rows))
    rng_local.shuffle(row_order)
    
    used = set()
    rows = [None] * n_rows
    
    for row_idx in row_order:
        # Try multiple times to build this row
        success = False
        for trial in range(10):
            row = _build_row_backtrack(k, target_scope, used, rng_local, trial)
            if row is not None:
                rows[row_idx] = row
                used |= _row_diffs(row)
                success = True
                break
        if not success:
            return None, None
    
    scope = max(max(r) for r in rows)
    if scope <= target_scope:
        return rows, scope
    return None, None


def _build_row_backtrack(k, max_val, used, rng, trial):
    """Build a row using backtracking with limited depth."""
    row = [0]
    
    def backtrack(pos, current_max):
        if pos == k + 1:
            return True
        # Try candidates in random order biased toward smaller
        candidates = []
        for x in range(current_max + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                candidates.append(x)
        
        if not candidates:
            return False
        
        # Shuffle with bias
        rng.shuffle(candidates)
        # Put smaller values earlier with some probability
        for i in range(len(candidates)):
            if rng.random() < 0.7:
                # Try smaller values first
                pass
        
        for c in candidates:
            row.append(c)
            new_used = {c - a for a in row[:-1]}
            if backtrack(pos + 1, c):
                return True
            row.pop()
        
        return False
    
    if backtrack(1, 0):
        return row
    return None


def _compact_rows(rows, k, max_scope, rng):
    """Try to reduce the scope of existing rows by shifting/compressing."""
    if rows is None:
        return None, None
    
    # Compute used differences
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    
    # Try to rebuild each row with smaller max
    new_rows = [None] * len(rows)
    
    for i, row in enumerate(rows):
        old_diffs = _row_diffs(row)
        # Remove this row's diffs from used
        remaining_used = used - old_diffs
        
        # Try to build a new row with smaller max
        best_new_row = None
        for trial in range(20):
            new_row = _build_row_randomized(k, max_scope, remaining_used, rng)
            if new_row is not None:
                scope = max(new_row)
                if best_new_row is None or scope < max(best_new_row):
                    best_new_row = new_row
        
        if best_new_row is not None:
            new_rows[i] = best_new_row
            remaining_used |= _row_diffs(best_new_row)
        else:
            # Keep original
            new_rows[i] = row
    
    scope = max(max(r) for r in new_rows)
    if scope < max_scope:
        return new_rows, scope
    return None, None

# EVOLVE-BLOCK-END

