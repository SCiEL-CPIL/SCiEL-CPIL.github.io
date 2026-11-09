import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Initialize rows using the greedy method (provides a valid baseline)
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
    
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Prepare fast validity checking structures
    # diff_counts[d] = number of rows containing difference d
    # row_diffs[r] = set of differences in row r
    diff_counts = {}
    row_diffs = [set() for _ in range(n_rows)]
    
    for r_idx, row in enumerate(best_rows):
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                d = row[j] - row[i]
                row_diffs[r_idx].add(d)
                diff_counts[d] = diff_counts.get(d, 0) + 1

    def is_valid_move(r_idx, pos, new_val, row):
        """
        Check if changing row[r_idx][pos] to new_val maintains validity.
        Returns True if valid, False otherwise.
        """
        # 1. Check bounds
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < k and new_val >= row[pos + 1]:
            return False
        
        # 2. Check for duplicate differences
        # Differences involving 'pos' are affected.
        # Old diffs: row[pos] - row[i] for i < pos, and row[j] - row[pos] for j > pos
        # New diffs: new_val - row[i] for i < pos, and row[j] - new_val for j > pos
        
        # We need to check if any NEW difference already exists in the global pool
        # (with count >= 1, considering that the old difference might be the one we're removing).
        # Actually, simpler: 
        # The set of differences in the row changes.
        # Let's identify which differences are removed and which are added.
        
        # Since we only change one element, the differences between other pairs remain.
        # Only differences involving index 'pos' change.
        
        # For each i < pos:
        #   Old d_old = row[pos] - row[i]
        #   New d_new = new_val - row[i]
        #   We are removing d_old and adding d_new.
        #   Validity condition: d_new must not be in diff_counts (excluding the d_old we are removing if d_new == d_old, but d_new != d_old since new_val != row[pos]).
        #   Wait, if d_new is in diff_counts, it could be from another row, or from this row (if it's a difference between two other elements).
        #   If it's from this row and not involving 'pos', it's still there. So diff_counts[d_new] must be 0.
        
        # For each j > pos:
        #   Old d_old = row[j] - row[pos]
        #   New d_new = row[j] - new_val
        #   Same logic.
        
        # Optimization: 
        # We can just simulate the changes on a local copy of diff_counts? No, too slow.
        # We can check:
        # For all i < pos:
        #   d_new = new_val - row[i]
        #   if diff_counts.get(d_new, 0) > 0: return False
        # For all j > pos:
        #   d_new = row[j] - new_val
        #   if diff_counts.get(d_new, 0) > 0: return False
        
        # Is this sufficient?
        # Yes, because the only differences that *change* are those involving 'pos'.
        # The differences NOT involving 'pos' are unchanged, so they remain valid (since we started valid).
        # The new differences MUST be distinct from ALL existing differences in the system.
        # Since we are removing the old ones, the only potential conflict is if a new difference matches:
        #   1. A difference in another row (count >= 1 in diff_counts)
        #   2. A difference in the same row that does NOT involve 'pos' (count >= 1 in diff_counts)
        #   3. Another new difference introduced by the same move (e.g. d_new1 == d_new2). 
        #      Wait, if d_new1 == d_new2, then when we check d_new1, diff_counts is 0 (initially), but we haven't added it yet.
        #      We need to check for collisions among the new differences themselves.
        
        # Let's refine:
        # Collect all new differences.
        new_diffs = []
        for i in range(pos):
            new_diffs.append(new_val - row[i])
        for j in range(pos + 1, k + 1):
            new_diffs.append(row[j] - new_val)
            
        # Check for internal collisions in new_diffs
        if len(set(new_diffs)) != len(new_diffs):
            return False
            
        # Check against global diff_counts
        for d in new_diffs:
            if diff_counts.get(d, 0) > 0:
                return False
                
        return True

    def apply_move(r_idx, pos, new_val, row):
        """Update diff_counts and row_diffs after a valid move."""
        # Remove old diffs
        for i in range(pos):
            d = row[pos] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
            row_diffs[r_idx].remove(d)
        for j in range(pos + 1, k + 1):
            d = row[j] - row[pos]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
            row_diffs[r_idx].remove(d)
            
        # Add new diffs
        for i in range(pos):
            d = new_val - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1
            row_diffs[r_idx].add(d)
        for j in range(pos + 1, k + 1):
            d = row[j] - new_val
            diff_counts[d] = diff_counts.get(d, 0) + 1
            row_diffs[r_idx].add(d)

    def revert_move(r_idx, pos, old_val, new_val, row):
        """Revert a move (essentially apply_move with old and new swapped, but careful with state)."""
        # It's safer to just call apply_move with the values swapped? 
        # No, apply_move assumes current state has 'new_val' and wants to go to 'old_val'.
        # But the row array currently has 'new_val'.
        # So we can just call apply_move(r_idx, pos, old_val, row) and then set row[pos] = old_val.
        # Wait, apply_move updates diff_counts based on current row values.
        # If row[pos] is new_val, and we call apply_move(..., old_val, ...), it will remove diffs involving new_val and add diffs involving old_val.
        # This is exactly what we want.
        apply_move(r_idx, pos, old_val, row)
        row[pos] = old_val

    # 3. Directed Local Search
    # We want to minimize scope. 
    # Strategy: Try to reduce the largest element in each row.
    # Or more generally, try to reduce any element if it doesn't increase the max and maintains validity.
    # Actually, reducing an element usually reduces the max if it IS the max.
    # Let's focus on reducing the maximum element of each row first.
    
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    
    # Sort rows by their max element in descending order to tackle the hardest first? 
    # Or just iterate.
    
    improved = True
    while time.time() < deadline and improved:
        improved = False
        # Shuffle row order to avoid bias
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        for r_idx in row_order:
            row = current_rows[r_idx]
            # Try to reduce the last element (which is the max)
            # And potentially other elements if it helps reduce the max later?
            # For now, just try to reduce the last element.
            # If we can reduce row[k], scope might decrease.
            
            # Try values from current max - 1 down to some limit
            # But we must maintain increasing order.
            # min_val = row[k-1] + 1
            
            # We can try a few random values or a range.
            # Let's try a window of values around the current max.
            # Or just try to reduce it by 1, 2, ... up to some amount.
            
            # More robust: try to find ANY valid value < row[k] that is > row[k-1]
            # Since we want to minimize, we try the smallest possible? No, that might block other rows.
            # Let's try values in descending order from row[k]-1 down to row[k-1]+1.
            # If we find one, we take it. This is greedy.
            
            current_max = row[k]
            for new_val in range(current_max - 1, row[k-1], -1):
                if is_valid_move(r_idx, k, new_val, row):
                    # Apply move
                    old_val = row[k]
                    apply_move(r_idx, k, new_val, row)
                    row[k] = new_val
                    improved = True
                    # Update scope
                    new_scope = max(max(r) for r in current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                        best_scope = new_scope
                    break # Found a reduction, move to next row
            
            # Also try to reduce other elements if it doesn't increase max?
            # This is harder to check quickly. 
            # For now, stick to reducing the max.
            # If we get stuck, we might need to move a non-max element to allow max reduction.
            # That requires a more complex search (e.g. swapping or 2-step moves).
            
            # Let's add a simple "shake" if stuck for a while? 
            # No, time limit.
            
    return best_rows
