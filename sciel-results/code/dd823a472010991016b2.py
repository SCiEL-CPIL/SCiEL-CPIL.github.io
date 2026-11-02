"""Difference triangle set (7,5): rows of k+1 increasing ints from 0, all pairwise diffs distinct across rows.
Strategy: randomized greedy construction + local search (swap marks to reduce max) + time-bounded restarts.
"""
import time
import random

_TIME_LIMIT = 140.0  # safety margin under 150s kill


def _row_diffs(row):
    """Set of positive differences within a row."""
    d = set()
    n = len(row)
    for i in range(n):
        a = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - a)
    return d


def _all_diffs(rows):
    """Set of all positive differences across all rows."""
    d = set()
    for r in rows:
        d |= _row_diffs(r)
    return d


def _is_valid(rows):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            a = r[i]
            for j in range(i + 1, n):
                d = r[j] - a
                if d in seen:
                    return False
                seen.add(d)
    return True


def _max_val(rows):
    return max(max(r) for r in rows)


def _greedy_row(k, used, rng):
    """Build one row of k+1 increasing ints starting at 0, avoiding used differences.
    Uses greedy with small randomization for diversity."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        # Check if x is valid: all diffs x - a for a in row must not be in used
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row


def _greedy_construct(n_rows, k, rng, seed):
    """Full greedy construction with randomization."""
    random.seed(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(k, used, rng)
        rows.append(row)
    return rows


def _local_search(rows, n_rows, k, rng, max_iters=5000):
    """Try to reduce the maximum value by swapping or removing marks.
    A swap: pick a row, pick two positions, try replacing one mark with a smaller unused value.
    Actually, we try: for each row, for each position (except first), try to replace with a smaller
    valid value that keeps all diffs distinct.
    """
    best_rows = [list(r) for r in rows]
    best_max = _max_val(best_rows)
    
    for it in range(max_iters):
        improved = False
        # Randomize row order
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for ri in row_order:
            row = best_rows[ri]
            n = len(row)
            # Try positions from the end (largest marks first)
            for pos in range(n - 1, 0, -1):
                old_val = row[pos]
                # Try to find a smaller value that works
                # The value must be > row[pos-1] and < old_val
                # And all new diffs must not conflict with used diffs from other rows + own row
                lo = row[pos - 1] + 1
                hi = old_val - 1
                if hi < lo:
                    continue
                
                # Compute used diffs excluding this row's contribution
                other_diffs = set()
                for rj in range(n_rows):
                    if rj == ri:
                        continue
                    other_diffs |= _row_diffs(best_rows[rj])
                
                # Diffs within this row excluding those involving position pos
                own_diffs_excl = set()
                for i in range(n):
                    if i == pos:
                        continue
                    a = row[i]
                    for j in range(n):
                        if j == pos or j <= i:
                            continue
                        own_diffs_excl.add(row[j] - a)
                
                forbidden = other_diffs | own_diffs_excl
                
                # Try values from hi down to lo (prefer smaller to reduce max)
                # But we want to reduce the global max, so try values that are smaller
                found = False
                for trial in range(hi, lo - 1, -1):
                    # Check: trial > row[pos-1] (ensured by range)
                    # New diffs involving position pos:
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if trial > row[i]:
                            new_diffs.add(trial - row[i])
                        else:
                            new_diffs.add(row[i] - trial)
                    
                    if not new_diffs & forbidden:
                        row[pos] = trial
                        improved = True
                        found = True
                        break
                
                if found:
                    break
        
        if not improved:
            break
    
    new_max = _max_val(best_rows)
    if new_max < best_max:
        best_max = new_max
    
    return best_rows, best_max


def _construct_with_budget(n_rows, k, time_limit):
    """Main construction with time budget."""
    start_time = time.time()
    rng = random.Random(42)
    
    # First: deterministic greedy to get a baseline
    used = set()
    baseline_rows = []
    for _ in range(n_rows):
        row = _greedy_row(k, used, rng)
        baseline_rows.append(row)
    
    best_rows = [list(r) for r in baseline_rows]
    best_max = _max_val(best_rows)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.3:
        return best_rows
    
    # Now try randomized constructions + local search
    attempt = 0
    while time.time() - start_time < time_limit - 5:
        attempt += 1
        # Randomized greedy: shuffle the order of difference selection
        seed = attempt * 7919 + 13
        rng2 = random.Random(seed)
        
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
        
        cur_max = _max_val(rows)
        
        # Local search to try to reduce
        rows, cur_max = _local_search(rows, n_rows, k, rng2, max_iters=2000)
        
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
        
        # Check time
        if time.time() - start_time > time_limit - 10:
            break
    
    return best_rows


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all positive differences distinct across rows."""
    start_time = time.time()
    
    try:
        result = _construct_with_budget(n_rows, k, _TIME_LIMIT)
    except Exception:
        # Fallback to simple greedy
        used = set()
        result = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            result.append(row)
    
    # Final validity check
    if not _is_valid(result):
        # Fallback to simple greedy
        used = set()
        result = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            result.append(row)
    
    return result
