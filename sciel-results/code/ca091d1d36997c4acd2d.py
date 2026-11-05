import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Difference triangle set: n_rows rows of k+1 increasing integers starting at 0,
    all positive within-row differences distinct across all rows.
    Minimize the maximum entry.
    """
    deadline = time.time() + 115  # 115s safety margin (limit is 120s)
    
    # Number of differences per row: C(k+1, 2)
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple counting bound: need total_diffs distinct positive differences,
    # each <= max_entry. So max_entry >= total_diffs (since diffs are 1..max_entry).
    # Actually diffs can be up to max_entry, and there are max_entry possible values.
    # So max_entry >= total_diffs.
    lower_bound = total_diffs
    
    # Try to find a solution with max_entry close to lower_bound
    # Start with a greedy solution to get an initial upper bound
    best_rows = _greedy_construct(n_rows, k)
    best_max = _get_max_entry(best_rows)
    
    if time.time() > deadline - 5:
        return best_rows
    
    # Simulated annealing
    rows, final_max = _simulated_annealing(n_rows, k, best_rows, best_max, deadline)
    
    if final_max < best_max:
        best_rows = rows
        best_max = final_max
    
    return best_rows


def _get_max_entry(rows):
    return max(max(row) for row in rows)


def _greedy_construct(n_rows, k):
    """Greedy construction: each row starts at 0, pick smallest next mark."""
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


def _row_diffs(row):
    """Get all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _simulated_annealing(n_rows, k, init_rows, init_max, deadline):
    """
    Simulated annealing to minimize the max entry.
    State: list of rows.
    Move: pick a random row, randomly perturb one of its entries (shift it),
    or swap two entries' positions in the difference structure.
    
    Actually, a better representation: think of each row as a set of k+1 points.
    We want to minimize the maximum point while keeping all differences unique.
    
    Strategy: Start from greedy. Try to reduce the max entry by:
    1. Picking a row with the largest entry
    2. Trying to replace its last element with a smaller one
    3. If that creates a conflict, try swapping differences or adjusting other rows
    
    Simpler approach: Local search where we try to "compress" rows.
    """
    import math
    
    rows = [list(r) for r in init_rows]
    current_max = init_max
    best_rows = [list(r) for r in rows]
    best_max = current_max
    
    # Track used differences
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    def get_conflict_count():
        """Count number of duplicate differences (0 = valid)."""
        all_d = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_d.append(row[j] - row[i])
        return len(all_d) - len(set(all_d))
    
    def try_reduce_max():
        """Try to reduce the maximum entry by 1 or more."""
        nonlocal rows, used_diffs, current_max, best_rows, best_max
        
        # Find rows with the current max
        max_val = max(max(r) for r in rows)
        max_rows = [i for i, r in enumerate(rows) if max(r) == max_val]
        
        if not max_rows:
            return False
        
        # Try to reduce the last element of a max row
        for ri in max_rows:
            row = rows[ri]
            old_last = row[-1]
            
            # Try replacing last element with smaller value
            for new_last in range(old_last - 1, max(0, old_last - 20), -1):
                if new_last <= row[-2]:
                    continue
                # Check if new differences conflict
                new_diffs = {new_last - a for a in row[:-1]}
                # Remove old differences involving old_last
                old_diffs = {old_last - a for a in row[:-1]}
                
                # Check conflicts: new_diffs should not be in (used_diffs - old_diffs)
                remaining_used = used_diffs - old_diffs
                if new_diffs & remaining_used:
                    continue
                # Also check within-row: new_last - a for a in row[:-1] are already unique
                # since row[:-1] is strictly increasing and new_last > row[-2]
                if len(new_diffs) != len(row) - 1:
                    continue
                
                # Accept
                row[-1] = new_last
                used_diffs -= old_diffs
                used_diffs |= new_diffs
                current_max = max(max(r) for r in rows)
                return True
        
        # Try adjusting non-last elements
        for ri in max_rows:
            row = rows[ri]
            for pos in range(1, len(row) - 1):
                old_val = row[pos]
                # Try decreasing
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    if new_val <= row[pos - 1]:
                        continue
                    # Check differences
                    # Differences involving position pos:
                    # With earlier: new_val - row[i] for i < pos
                    # With later: row[j] - new_val for j > pos
                    new_diffs_earlier = {new_val - row[i] for i in range(pos)}
                    new_diffs_later = {row[j] - new_val for j in range(pos + 1, len(row))}
                    new_diffs = new_diffs_earlier | new_diffs_later
                    
                    old_diffs_earlier = {old_val - row[i] for i in range(pos)}
                    old_diffs_later = {row[j] - old_val for j in range(pos + 1, len(row))}
                    old_diffs = old_diffs_earlier | old_diffs_later
                    
                    remaining_used = used_diffs - old_diffs
                    if new_diffs & remaining_used:
                        continue
                    if len(new_diffs) != len(new_diffs_earlier) + len(new_diffs_later):
                        continue
                    
                    row[pos] = new_val
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    current_max = max(max(r) for r in rows)
                    return True
        
        return False
    
    # Iterative improvement with randomization
    attempts = 0
    max_attempts = 500000
    random.seed(42)
    
    while time.time() < deadline - 2:
        attempts += 1
        
        # Try deterministic reduction
        if try_reduce_max():
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in rows]
            continue
        
        # If stuck, try a random perturbation followed by reduction
        if attempts % 100 == 0:
            # Random perturbation: pick a random row and a random position,
            # increase the value slightly to create space, then try to reduce others
            ri = random.randint(0, n_rows - 1)
            pos = random.randint(1, len(rows[ri]) - 1)
            old_val = rows[ri][pos]
            
            # Remove old diffs
            old_diffs = set()
            for i in range(len(rows[ri])):
                if i != pos:
                    d = abs(rows[ri][pos] - rows[ri][i])
                    old_diffs.add(d)
            
            # Try a new value
            new_val = old_val + random.randint(1, 5)
            if pos < len(rows[ri]) - 1 and new_val >= rows[ri][pos + 1]:
                continue
            if pos > 0 and new_val <= rows[ri][pos - 1]:
                continue
            
            new_diffs = set()
            for i in range(len(rows[ri])):
                if i != pos:
                    d = abs(new_val - rows[ri][i])
                    new_diffs.add(d)
            
            remaining_used = used_diffs - old_diffs
            if new_diffs & remaining_used:
                continue
            if len(new_diffs) != len(rows[ri]) - 1:
                continue
            
            rows[ri][pos] = new_val
            used_diffs -= old_diffs
            used_diffs |= new_diffs
            current_max = max(max(r) for r in rows)
            
            # Now try to reduce
            for _ in range(10):
                if not try_reduce_max():
                    break
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in rows]
        
        if attempts % 10000 == 0:
            # Periodic check
            pass
    
    return best_rows, best_max


# EVOLVE-BLOCK-END
