import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _validate_full(rows):
    """
    Check if the set of rows is a valid DTS.
    Returns True if valid, False otherwise.
    """
    used = set()
    for row in rows:
        if len(row) != k_global + 1:
            return False
        if row[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        # Check differences
        rd = _row_diffs(row)
        if len(rd) != len(row) * (len(row) - 1) // 2:
            return False # Internal collision
        if used & rd:
            return False # Cross-row collision
        used |= rd
    return True

def _backtrack_construct(n_rows, k, max_val, time_limit=10.0):
    """
    Try to construct a valid DTS with max entry <= max_val.
    Returns list of rows if found, None if not found or timeout.
    """
    start_time = time.time()
    
    # We build rows one by one.
    # State: list of current rows, set of used differences.
    rows = []
    used_diffs = set()
    
    # To improve search, we can sort rows or use symmetry breaking.
    # For now, simple backtracking.
    
    def backtrack(row_idx):
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            return True
            
        # Try to construct row row_idx
        # The row must start with 0.
        # We need k more elements.
        
        # We can use a recursive helper to build the row.
        def build_row(current_row, pos):
            if time.time() - start_time > time_limit:
                return False
            
            if pos == k + 1:
                # Check if this row is valid with respect to used_diffs
                rd = _row_diffs(current_row)
                if rd & used_diffs:
                    return False
                # If valid, commit
                rows.append(current_row[:])
                used_diffs.update(rd)
                
                if backtrack(row_idx + 1):
                    return True
                
                # Undo
                rows.pop()
                used_diffs.difference_update(rd)
                return False
            
            # Determine range for next element
            if pos == 0:
                current_row.append(0)
                if build_row(current_row, pos + 1):
                    return True
                current_row.pop()
                return False
            else:
                prev_val = current_row[pos - 1]
                # Max value is max_val
                # Also, to ensure strictly increasing, next > prev_val
                # We can prune: if current_row[pos-1] + remaining_needed > max_val, fail?
                # Not exactly, but we can bound the search.
                
                # Heuristic: Try values close to the minimum first? Or spread out?
                # For DTS, "tight" packing is often good.
                # Let's iterate from prev_val + 1 to max_val.
                
                # Optimization: If we are at the last element (pos == k),
                # we just need to pick a value such that no new diffs collide.
                
                for val in range(prev_val + 1, max_val + 1):
                    current_row.append(val)
                    
                    # Quick check: internal diffs of current_row so far
                    # and potential diffs with new val
                    # Actually, just checking full row diffs at the end is safe but slow.
                    # Let's do incremental check.
                    
                    # Check diffs involving val
                    valid = True
                    new_diffs = set()
                    for i in range(pos):
                        d = val - current_row[i]
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Check against global used_diffs
                        # Note: used_diffs contains diffs from previous rows.
                        # It does NOT contain diffs from the current row yet.
                        # So we only need to check new_diffs against used_diffs.
                        # Internal consistency of current row is handled by new_diffs uniqueness.
                        if not (new_diffs & used_diffs):
                            if build_row(current_row, pos + 1):
                                return True
                            
                    current_row.pop()
                
                return False
        
        # Start building row
        if build_row([], 0):
            return True
        
        return False

    if backtrack(0):
        return rows
    return None

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
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

def _sa_polish(rows, n_rows, k, time_limit=10.0):
    """
    Simulated Annealing to minimize the global max.
    """
    start_time = time.time()
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    row_maxes = [max(r) for r in current_rows]
    all_used = _all_used(current_rows)
    row_diffs = [_row_diffs(r) for r in current_rows]
    
    T = 50.0
    T_min = 0.1
    cool = 0.9999
    
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates
        candidates = []
        
        # Decreasing
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(20, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
            
        # Increasing
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
        max_increase = min(10, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
            
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            # Validate
            new_diffs = set()
            valid = True
            n = len(cand_row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = cand_row[j] - cand_row[i]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
                
            # Check cross-row
            for d in new_diffs:
                if d in all_used and d not in row_diffs[ri]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Acceptance
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
                        
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                moved = True
                break
                
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
    return best_rows

def construct(n_rows, k):
    global k_global
    k_global = k
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Try Backtracking with increasing scope
    # Estimate lower bound
    # Counting bound: n_rows * k * (k+1) / 2 distinct diffs?
    # Total diffs needed: n_rows * k * (k+1) / 2
    # Max diff in scope M is M.
    # So M >= total_diffs / n_rows ? No, diffs are shared.
    # Simple bound: M >= k * (k+1) / 2 ?
    
    # Let's start from a reasonable guess and go up.
    # For (7,5), best is 112.
    # Let's try to find a solution with scope < 112.
    
    best_rows = None
    best_val = float('inf')
    
    # Greedy init to have a fallback
    greedy_rows = _greedy_init(n_rows, k)
    greedy_val = _global_max(greedy_rows)
    best_rows = [r[:] for r in greedy_rows]
    best_val = greedy_val
    
    # Try backtracking with scopes starting from lower bound
    # Lower bound estimate
    lb = 1
    # A better lower bound:
    # Each row has k(k+1)/2 diffs.
    # Total diffs = n_rows * k(k+1)/2.
    # These must be distinct and <= M.
    # So M >= n_rows * k(k+1)/2.
    # For 7, 5: 7 * 15 = 105.
    lb = n_rows * k * (k + 1) // 2
    
    # Try scopes from lb to lb + 20
    # We give backtracking a time limit for each scope attempt.
    # If we find a solution, we try to polish it.
    
    # To save time, we can run backtracking in a loop with increasing M.
    # But backtracking is expensive.
    # Let's try a few specific M values.
    
    # If time allows, try to find the minimal M.
    # Given 150s limit, we should be careful.
    
    # Let's try M = lb, lb+1, ... up to lb+10 with 5s limit each?
    # That's 50s.
    # Then use remaining time for SA on the best found.
    
    bt_time_limit = 5.0
    for m in range(lb, lb + 15):
        if time.time() > deadline - 20.0:
            break
            
        rows = _backtrack_construct(n_rows, k, m, time_limit=bt_time_limit)
        if rows:
            val = _global_max(rows)
            if val < best_val:
                best_val = val
                best_rows = [r[:] for r in rows]
            # If we found one, we might want to try smaller?
            # No, we are increasing m. The first valid m is the minimal for this search order.
            # But backtracking might find a valid set with max entry < m even if m is larger?
            # No, if we constrain max entry <= m, and find a solution, its max entry is <= m.
            # Since we start from lb, the first success is likely close to optimal.
            # We can break here if we want to save time for SA.
            # Or continue to see if we can find a "better" packing?
            # The objective is min max entry.
            # So the first valid m is the best we can get with this method.
            # Let's break and use SA to try to reduce it further?
            # SA can reduce max entry if there's slack.
            break
            
    # If backtracking didn't find anything better than greedy, use greedy.
    # If it found something, use it.
    
    # 2. SA Polish
    # Use remaining time for SA.
    sa_time = max(1.0, deadline - time.time() - 5.0)
    if sa_time > 1.0:
        polished_rows = _sa_polish(best_rows, n_rows, k, time_limit=sa_time)
        if _global_max(polished_rows) < best_val:
            best_rows = polished_rows
            best_val = _global_max(best_rows)
            
    return best_rows

# EVOLVE-BLOCK-END
