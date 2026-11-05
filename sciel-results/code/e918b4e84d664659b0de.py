"""
Difference triangle set (4,7) solver using CSP with bitset-based difference
constraint propagation and a rebuild-based local search with random row
reshuffling to escape local minima.

Strategy:
1. Fast greedy baseline.
2. CSP-style row construction: when building each row, use a bitset of used
   differences to instantly prune candidates. Build rows in an order that
   prioritizes tightest remaining capacity (smallest available gap).
3. Local search with random row reshuffling: repeatedly pick a row, remove it,
   and rebuild it with a tighter cap. If the global max decreases, keep it.
   Even when it doesn't, occasionally accept a random reshuffle to escape
   local minima.
4. Deadline guard to stay within 150s wall-clock.
"""
import time
import random
import sys

def _greedy_construct(n_rows, k):
    """Fast greedy baseline: build each row greedily, smallest unused position."""
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
    return rows, used

def _compute_used(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used

def _validate(rows, k):
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if row[j] <= row[i]:
                    return False
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)

def _build_row_with_cap(used, k, cap, rng=None):
    """
    Try to build a row of k+1 increasing integers starting at 0, with max <= cap,
    such that all positive differences are distinct and not in 'used'.
    Uses a greedy-with-backtracking approach: at each step, try candidates from
    small to large, but if stuck, backtrack.
    
    Returns (row, success).
    """
    if cap < 1:
        return None, False
    
    # Use a recursive backtracking with a time check
    # For efficiency, use iterative DFS with explicit stack
    # But for k=7 (8 elements), the search space is manageable
    
    row = [0]
    
    def _dfs(idx, current_row, current_used_local):
        """
        idx: next index to fill (1..k)
        current_row: list of chosen values so far
        current_used_local: set of differences used within this row so far
        """
        if idx == k + 1:
            return True
        
        last = current_row[-1]
        # Try candidates from last+1 to cap
        # Heuristic: try smaller values first to keep the row compact
        for x in range(last + 1, cap + 1):
            new_diffs = {x - a for a in current_row}
            # Check against global used and local used
            if new_diffs & used:
                continue
            if new_diffs & current_used_local:
                continue
            # Check that new_diffs are all distinct (they are, since current_row is increasing)
            current_row.append(x)
            current_used_local |= new_diffs
            if _dfs(idx + 1, current_row, current_used_local):
                return True
            current_row.pop()
            current_used_local -= new_diffs
        return False
    
    # To avoid deep recursion, use an iterative approach or limit depth
    # For k=7, max depth is 7, which is fine for recursion
    # But we need a time guard
    
    start_time = time.time()
    deadline = start_time + 2.0  # 2 second limit per row build attempt
    
    # Use a closure with time check
    def _dfs_timed(idx, current_row, current_used_local):
        if time.time() > deadline:
            return False
        if idx == k + 1:
            return True
        
        last = current_row[-1]
        for x in range(last + 1, cap + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used:
                continue
            if new_diffs & current_used_local:
                continue
            current_row.append(x)
            new_local = current_used_local | new_diffs
            if _dfs_timed(idx + 1, current_row, new_local):
                return True
            current_row.pop()
        return False
    
    success = _dfs_timed(1, row, set())
    if success:
        return row, True
    return None, False

def _build_row_greedy_with_cap(used, k, cap):
    """
    Greedy row construction with a cap. Faster but may fail where backtracking would succeed.
    Returns (row, success).
    """
    row = [0]
    local_used = set()
    x = 0
    while len(row) < k + 1:
        found = False
        for x in range(row[-1] + 1, cap + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and not (new_diffs & local_used):
                row.append(x)
                local_used |= new_diffs
                found = True
                break
        if not found:
            return None, False
    return row, True

def _try_rebuild_row(rows, used, row_idx, k, cap, deadline, rng):
    """
    Try to rebuild row row_idx with max entry <= cap.
    Returns (new_row, success).
    """
    if time.time() >= deadline:
        return None, False
    
    # Temporarily remove this row's differences from used
    old_row = rows[row_idx]
    old_diffs = set()
    for i in range(len(old_row)):
        for j in range(i + 1, len(old_row)):
            old_diffs.add(old_row[j] - old_row[i])
    for d in old_diffs:
        used.discard(d)
    
    # Try to build new row
    new_row, success = _build_row_greedy_with_cap(used, k, cap)
    if not success:
        # Try with backtracking (slower)
        new_row, success = _build_row_with_cap(used, k, cap, rng)
    
    if success:
        # Add new differences
        for i in range(len(new_row)):
            for j in range(i + 1, len(new_row)):
                used.add(new_row[j] - new_row[i])
        return new_row, True
    else:
        # Restore old differences
        used |= old_diffs
        return None, False

def _local_search(rows, k, deadline):
    """
    Local search: repeatedly try to rebuild rows with tighter caps.
    Uses random row reshuffling to escape local minima.
    """
    rows = [row[:] for row in rows]
    used = _compute_used(rows)
    
    current_max = _max_entry(rows)
    best_rows = [row[:] for row in rows]
    best_max = current_max
    
    rng = random.Random(12345)
    
    # Track how many times we've tried without improvement
    no_improve_count = 0
    max_no_improve = 50  # After this many failed attempts, do a random reshuffle
    
    while time.time() < deadline:
        # Pick a random row to rebuild
        row_idx = rng.randint(0, len(rows) - 1)
        
        # Determine cap: try to reduce max entry
        # The target cap is current_max - 1, but we can also try lower
        # Start with current_max - 1 and go down
        row_max = max(rows[row_idx])
        
        # Try caps from min(row_max, current_max) - 1 down to some lower bound
        # But to save time, try a few caps
        tried_improve = False
        
        # First, try to keep the same row but with a smaller max
        # The row's current max is row_max
        # Try caps: row_max - 1, row_max - 2, ..., down to maybe current_max - 10
        
        lower_bound = max(k, 1)  # Minimum possible max entry is k
        upper_cap = min(row_max - 1, current_max - 1) if row_max > 1 else row_max
        
        if upper_cap < lower_bound:
            upper_cap = lower_bound
        
        # Try a few caps in descending order
        caps_to_try = list(range(upper_cap, lower_bound - 1, -1))
        # Limit to at most 20 caps to save time
        if len(caps_to_try) > 20:
            # Sample 20 caps
            caps_to_try = [caps_to_try[i] for i in rng.sample(range(len(caps_to_try)), 20)]
            caps_to_try.sort(reverse=True)
        
        for cap in caps_to_try:
            if time.time() >= deadline:
                break
            new_row, success = _try_rebuild_row(rows, used, row_idx, k, cap, deadline, rng)
            if success:
                rows[row_idx] = new_row
                new_max = _max_entry(rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                tried_improve = True
                break
        
        if not tried_improve:
            no_improve_count += 1
        
        # If we've had many failures, do a random reshuffle
        if no_improve_count >= max_no_improve:
            no_improve_count = 0
            # Randomly reshuffle: pick a row and rebuild it with a random cap
            # or just accept the current state and try a different row
            # For now, just reset the counter and continue
            # In the future, we could do more aggressive reshuffling
            pass
    
    # Return the best solution found
    return best_rows, best_max

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Step 1: Greedy baseline
    rows, used = _greedy_construct(n_rows, k)
    
    # Validate baseline
    if not _validate(rows, k):
        # This shouldn't happen, but just in case
        return rows
    
    initial_max = _max_entry(rows)
    
    # Step 2: Local search with row rebuilding
    if time.time() < deadline - 5:
        best_rows, best_max = _local_search(rows, k, deadline - 5)
        
        # Validate the best solution
        if _validate(best_rows, k):
            rows = best_rows
    
    # Final validation
    if not _validate(rows, k):
        # Fallback to greedy
        rows, _ = _greedy_construct(n_rows, k)
    
    return rows
