"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    s = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(r[j] - r[i])
    return s


def _valid(rows, k):
    """Check validity: all rows start at 0, strictly increasing, and all diffs distinct."""
    if len(rows) != k + 1:  # placeholder, actual check below
        pass
    used = set()
    for r in rows:
        if not r or r[0] != 0:
            return False
        for i in range(1, len(r)):
            if r[i] <= r[i - 1]:
                return False
        diffs = _row_diffs(r)
        if diffs & used:
            return False
        used |= diffs
    return True


def _greedy_row(used, k, max_val=None):
    """Build one row greedily: start at 0, pick smallest x such that all new diffs are unused."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used |= new_diffs
    return row


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes smallest available marks."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _random_row(used, k, max_val):
    """Try to build a random row within max_val that doesn't conflict with used diffs."""
    row = [0]
    x = 0
    attempts = 0
    while len(row) < k + 1:
        x += 1
        if x > max_val:
            return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used |= new_diffs
        else:
            # try a few random jumps
            for _ in range(3):
                x += random.randint(1, 5)
                if x > max_val:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    break
            else:
                continue
    return row


def _local_search_shrink(rows, k, used, deadline):
    """Try to shrink individual entries to reduce the max value."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(rows)):
            row = rows[ri]
            for pi in range(1, len(row)):
                old_val = row[pi]
                # Try to decrease row[pi]
                for new_val in range(old_val - 1, row[pi - 1], -1):
                    # Check if new_val is > row[pi-1] and strictly increasing
                    if new_val <= row[pi - 1]:
                        break
                    # Check that all diffs with earlier entries in this row are still valid
                    # and not in used (excluding this row's own diffs)
                    # First, remove this row's diffs from used
                    old_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            old_diffs.add(row[j] - row[i])
                    other_used = used - old_diffs
                    
                    # Build new row
                    new_row = row[:]
                    new_row[pi] = new_val
                    # Check strictly increasing
                    valid_inc = all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1))
                    if not valid_inc:
                        continue
                    # Check new diffs don't conflict
                    new_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            new_diffs.add(new_row[j] - new_row[i])
                    if new_diffs & other_used:
                        continue
                    # Accept
                    row[pi] = new_val
                    used = other_used | new_diffs
                    improved = True
                    break
    return rows


def _local_search_swap(rows, k, used, deadline):
    """Try swapping/shifting entries to reduce max."""
    # Try reducing the last element of each row
    for ri in range(len(rows)):
        if time.time() > deadline:
            break
        row = rows[ri]
        if len(row) < 2:
            continue
        old_last = row[-1]
        # Try to reduce the last element
        for new_last in range(old_last - 1, row[-2], -1):
            if new_last <= row[-2]:
                break
            # Remove this row's diffs
            old_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    old_diffs.add(row[j] - row[i])
            other_used = used - old_diffs
            
            new_row = row[:]
            new_row[-1] = new_last
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            if new_diffs & other_used:
                continue
            # Accept
            row[-1] = new_last
            used = other_used | new_diffs
            break
    return rows


def _reconstruct_rows(rows, k):
    """Rebuild used set from rows."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(r[j] - r[i])
    return used


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial greedy construction
    rows = _greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    
    # Local search: shrink entries
    used = _reconstruct_rows(rows, k)
    rows = _local_search_shrink(rows, k, used, deadline)
    scope = max(max(r) for r in rows)
    if scope < best_scope:
        best_scope = scope
        best_rows = [r[:] for r in rows]
    
    # Try multiple restarts with randomization
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Greedy with some randomness in row ordering
        random.seed(attempts * 7919 + 13)
        used = set()
        rows_try = []
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows_try.append(row)
        
        scope_try = max(max(r) for r in rows_try)
        if scope_try < best_scope:
            best_scope = scope_try
            best_rows = [r[:] for r in rows_try]
            # Polish
            used2 = _reconstruct_rows(rows_try, k)
            rows_try = _local_search_shrink(rows_try, k, used2, deadline)
            scope_try = max(max(r) for r in rows_try)
            if scope_try < best_scope:
                best_scope = scope_try
                best_rows = [r[:] for r in rows_try]
    
    # Final polish on best
    if time.time() < deadline - 2:
        used_best = _reconstruct_rows(best_rows, k)
        best_rows = _local_search_shrink(best_rows, k, used_best, deadline)
    
    return best_rows

# EVOLVE-BLOCK-END
