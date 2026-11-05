"""Improved program for difference triangle set (7,5) using bitmask and local search."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _compute_mask(row):
    """Compute the bitmask of all positive differences in the row."""
    mask = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            mask |= (1 << d)
    return mask

def _row_is_valid_internal(row):
    """Check if the row has distinct internal differences."""
    n = len(row)
    diffs = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _find_feasible_extension(row, used_mask, max_val_hint):
    """
    Find the smallest value > row[-1] that can be appended to row
    without creating duplicate internal differences or conflicting with used_mask.
    Returns (new_row, new_mask) or (None, None) if not found within reasonable bounds.
    """
    last_val = row[-1]
    # We don't know the exact upper bound, but we can search up to a limit.
    # For column-wise, we usually want small values.
    # Let's search up to last_val + 200 or so, or just keep going until found.
    # To be safe and fast, we'll check a range.
    
    # Pre-calculate existing diffs in the row to check internal validity quickly
    # Actually, we just need to check new diffs against old diffs and used_mask.
    
    for val in range(last_val + 1, last_val + 300): 
        new_diffs = []
        valid = True
        for prev in row:
            d = val - prev
            if d in new_diffs:
                valid = False
                break
            new_diffs.append(d)
        
        if not valid:
            continue
            
        # Check against used_mask
        conflict = False
        for d in new_diffs:
            if (used_mask >> d) & 1:
                conflict = True
                break
        
        if conflict:
            continue
            
        return val, new_diffs
    return None, None

def _construct_column_wise(n_rows, k, rng, deadline):
    """
    Construct rows column by column.
    """
    rows = [[0] for _ in range(n_rows)]
    used_mask = 0
    
    # We track the max value we've seen to set a soft limit for search
    current_max_val = 0
    
    for col in range(1, k + 1):
        # Shuffle row order to break symmetry
        order = list(range(n_rows))
        rng.shuffle(order)
        
        for r_idx in order:
            row = rows[r_idx]
            last_val = row[-1]
            
            val, new_diffs = _find_feasible_extension(row, used_mask, current_max_val)
            
            if val is None:
                # If we couldn't find a small extension, we might be stuck or need to go larger.
                # In column-wise, if one row fails, the whole construction fails for this random seed.
                # We could backtrack, but that's complex. Instead, we just return None for this attempt.
                return None
            
            row.append(val)
            for d in new_diffs:
                used_mask |= (1 << d)
                
            if val > current_max_val:
                current_max_val = val
                
    return rows

def _find_replacement_row(other_mask, k, max_scope_limit, rng, deadline):
    """
    Find a valid row of length k+1 starting at 0 such that:
    1. Internal differences are distinct.
    2. No difference conflicts with other_mask.
    3. Max entry <= max_scope_limit.
    Minimizes the max entry.
    """
    # We use a backtracking search to find the row with the smallest possible max.
    # Since k=5, the row has 6 elements.
    # We can iterate possible max values from lower bound up to max_scope_limit.
    
    # Lower bound for max entry:
    # We need 15 distinct differences.
    # The smallest possible differences are 1, 2, ..., 15.
    # But they must form a valid set for a row starting at 0.
    # A simple lower bound is the k-th triangular number? No.
    # Let's just start from a reasonable small number, e.g., 20, and go up.
    
    # Actually, we can just do a DFS with pruning.
    # State: current_row, current_mask
    # Goal: len(current_row) == k+1
    
    best_row = None
    best_max = max_scope_limit + 1
    
    # To speed up, we can try to build the row greedily with some randomization,
    # but constrained by max_scope_limit.
    
    # Let's try a few randomized greedy attempts with the constraint.
    for _ in range(50):
        row = [0]
        current_mask = 0
        success = True
        
        for step in range(k):
            # Find smallest val > row[-1] such that val <= max_scope_limit
            # and new diffs don't conflict with current_mask or other_mask
            last_val = row[-1]
            found = False
            # Search space is small if max_scope_limit is tight
            for val in range(last_val + 1, max_scope_limit + 1):
                new_diffs = [val - prev for prev in row]
                
                # Check internal uniqueness
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                    
                # Check conflict with current_mask
                valid = True
                for d in new_diffs:
                    if (current_mask >> d) & 1:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Check conflict with other_mask
                for d in new_diffs:
                    if (other_mask >> d) & 1:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Accept
                row.append(val)
                for d in new_diffs:
                    current_mask |= (1 << d)
                found = True
                break
                
            if not found:
                success = False
                break
                
        if success:
            max_val = row[-1]
            if max_val < best_max:
                best_max = max_val
                best_row = row[:]
                
            # If we found a very small one, we can stop early?
            # No, let's try a few more to see if we can get even smaller.
            if best_max <= 20: # Heuristic: 20 is quite small for k=5
                break
                
    return best_row

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    best_rows = None
    best_scope = float('inf')
    
    # 1. Quick Deterministic Baseline
    # Greedy: always pick the smallest possible value.
    rows = [[0] for _ in range(n_rows)]
    used_mask = 0
    valid = True
    for col in range(1, k + 1):
        for r_idx in range(n_rows):
            row = rows[r_idx]
            last_val = row[-1]
            val, new_diffs = _find_feasible_extension(row, used_mask, 0)
            if val is None:
                valid = False
                break
            row.append(val)
            for d in new_diffs:
                used_mask |= (1 << d)
        if not valid:
            break
            
    if valid:
        scope = max(r[-1] for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            
    # 2. Randomized Column-Wise Construction
    # Try multiple random orderings
    attempts = 0
    while time.time() < deadline - 10:
        rng = random.Random(attempts * 123 + 456)
        rows = _construct_column_wise(n_rows, k, rng, deadline)
        if rows:
            scope = max(r[-1] for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
        attempts += 1
        
    # 3. Local Search / Refinement
    if best_rows and time.time() < deadline - 5:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        rng = random.Random()
        
        # Calculate masks for each row
        row_masks = [_compute_mask(r) for r in current_rows]
        
        # Iteratively try to improve rows
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to improve each row
            for i in range(n_rows):
                if time.time() > deadline - 2:
                    break
                    
                # Calculate mask of all OTHER rows
                other_mask = 0
                for j in range(n_rows):
                    if j != i:
                        other_mask |= row_masks[j]
                        
                current_max = current_rows[i][-1]
                
                # Try to find a replacement with max < current_max
                # We can try to lower the bound slightly to see if we can find a better one
                # Or just try to find ANY valid row with max < current_max
                # Since we want to minimize scope, we care about the global max.
                # If current_rows[i] is not the global max, improving it might not help the score.
                # But it might free up space for other rows? No, masks are disjoint.
                # Actually, replacing a row doesn't affect the feasibility of others,
                # as long as the new mask is disjoint from others.
                # So we can independently minimize each row's max, given the others.
                
                # Let's try to find a row with max < current_max
                # We'll search from a lower bound up to current_max - 1
                # Lower bound is roughly 15 (sum of 1..5? No, diffs are 1..15 but they must fit in a row)
                # A row with max M has diffs in [1, M].
                # We need 15 distinct diffs.
                
                # Try to find a better row
                new_row = _find_replacement_row(other_mask, k, current_max - 1, rng, deadline)
                
                if new_row:
                    new_max = new_row[-1]
                    if new_max < current_max:
                        current_rows[i] = new_row
                        row_masks[i] = _compute_mask(new_row)
                        improved = True
                        
        # Update best if we improved the global scope
        final_scope = max(r[-1] for r in current_rows)
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = current_rows

    if best_rows is None:
        # Fallback: return the deterministic baseline if it was valid, or a simple valid set
        # The deterministic baseline was stored in best_rows if valid.
        # If not, we need a fallback.
        # A simple fallback:
        # Just use a large spread.
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                diffs = {x - a for a in row}
                if not diffs & used:
                    row.append(x)
                    used |= diffs
                x += 1
            rows.append(row)
        return rows
        
    return best_rows

# EVOLVE-BLOCK-END
