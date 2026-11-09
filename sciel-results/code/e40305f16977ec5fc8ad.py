"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences within a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)


def _greedy_build_row(used_diffs, k, max_val=None):
    """
    Greedily build a row of k+1 increasing integers starting at 0,
    choosing the smallest next mark whose differences to earlier marks
    are all unused. Returns the row, or None if stuck.
    """
    row = [0]
    x = 0
    limit = max_val if max_val is not None else 10000
    while len(row) < k + 1:
        x += 1
        if x > limit:
            return None
        new_diffs = set(x - a for a in row)
        if not new_diffs & used_diffs:
            row.append(x)
            used_diffs |= new_diffs
    return row


def _backtrack_build_row(used_diffs, k, max_val=200):
    """
    Backtracking search to build a tight row. Tries to find a row with
    smallest possible maximum element, given the set of already-used differences.
    """
    best_row = None
    best_max = max_val + 1
    
    def backtrack(row, used, current_max):
        nonlocal best_row, best_max
        if len(row) == k + 1:
            if row[-1] < best_max:
                best_max = row[-1]
                best_row = list(row)
            return
        
        # Pruning: if current_max already >= best_max, no point continuing
        if current_max >= best_max:
            return
        
        last = row[-1]
        # Try next values starting from last+1
        for x in range(last + 1, best_max):
            new_diffs = set(x - a for a in row)
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
            backtrack(row, used, x)
            row.pop()
            used -= new_diffs
    
    backtrack([0], set(used_diffs), 0)
    return best_row


def _local_search(rows, k, time_budget):
    """
    Local search: try to reduce the scope by rebuilding individual rows
    with backtracking, keeping the best valid configuration found.
    """
    start_time = time.time()
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    for iteration in range(200):
        if time.time() - start_time > time_budget:
            break
        
        # Try rebuilding each row
        for i in range(len(rows)):
            if time.time() - start_time > time_budget:
                break
            
            # Collect used diffs from all other rows
            used_diffs = set()
            for j, row in enumerate(rows):
                if j == i:
                    continue
                used_diffs |= _row_diffs(row)
            
            # Try to find a tighter row
            old_row = rows[i]
            old_max = old_row[-1]
            
            new_row = _backtrack_build_row(used_diffs, k, max_val=old_max)
            if new_row is not None and new_row[-1] < old_max:
                rows[i] = new_row
                # Check validity
                if _is_valid(rows):
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                else:
                    rows[i] = old_row
    
    return best_rows


def _randomized_rebuild(rows, k, used_diffs, row_idx, time_budget):
    """
    Try to rebuild a specific row using randomized greedy with restarts.
    """
    start_time = time.time()
    best_row = None
    best_max = rows[row_idx][-1]
    
    for attempt in range(50):
        if time.time() - start_time > time_budget:
            break
        
        row = [0]
        x = 0
        local_used = set(used_diffs)
        success = True
        
        while len(row) < k + 1:
            # Randomize the choice among valid next values
            candidates = []
            for trial_x in range(x + 1, x + 20):
                new_diffs = set(trial_x - a for a in row)
                if not new_diffs & local_used:
                    candidates.append(trial_x)
            
            if not candidates:
                # Try further out
                for trial_x in range(x + 20, x + 50):
                    new_diffs = set(trial_x - a for a in row)
                    if not new_diffs & local_used:
                        candidates.append(trial_x)
                        break
            
            if not candidates:
                success = False
                break
            
            # Pick randomly from candidates, biased toward smaller
            x = random.choice(candidates[:min(5, len(candidates))])
            row.append(x)
            local_used |= set(x - a for a in row[:-1])
        
        if success and row[-1] < best_max:
            best_max = row[-1]
            best_row = list(row)
    
    return best_row


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences
    within rows are distinct across all rows.
    """
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Phase 1: Initial greedy construction
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_build_row(used_diffs, k)
        if row is None:
            # Fallback: allow larger values
            row = _greedy_build_row(used_diffs, k, max_val=5000)
            if row is None:
                # Last resort: just use consecutive integers
                row = list(range(k + 1))
        rows.append(row)
    
    initial_scope = _scope(rows)
    
    if time.time() > deadline:
        return rows
    
    # Phase 2: Local search with backtracking row rebuilds
    search_time = min(60, max(5, (deadline - time.time()) * 0.5))
    improved_rows = _local_search(rows, k, search_time)
    
    if time.time() > deadline:
        return improved_rows
    
    # Phase 3: Randomized rebuild attempts for further improvement
    remaining_time = deadline - time.time() - 5
    if remaining_time > 2:
        best_rows = [list(r) for r in improved_rows]
        best_scope = _scope(best_rows)
        
        while time.time() < deadline - 5:
            # Pick a random row to rebuild
            row_idx = random.randint(0, n_rows - 1)
            
            # Collect used diffs from other rows
            used_diffs = set()
            for j, row in enumerate(best_rows):
                if j == row_idx:
                    continue
                used_diffs |= _row_diffs(row)
            
            new_row = _randomized_rebuild(best_rows, k, used_diffs, row_idx, 
                                          min(2, deadline - time.time() - 5))
            if new_row is not None and new_row[-1] < best_rows[row_idx][-1]:
                old_row = best_rows[row_idx]
                best_rows[row_idx] = new_row
                if _is_valid(best_rows):
                    new_scope = _scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                else:
                    best_rows[row_idx] = old_row
    
    # Final validity check
    if not _is_valid(best_rows):
        # Fallback to initial greedy result
        return rows
    
    return best_rows

# EVOLVE-BLOCK-END
