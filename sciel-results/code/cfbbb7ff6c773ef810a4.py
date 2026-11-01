import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # 1. Initial Greedy Construction
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # Helper to check validity of a move
    def can_move(r, j, new_val):
        if new_val <= rows[r][j-1] or new_val >= rows[r][j]:
            return False
        # Calculate old differences involving index j
        old_diffs = set()
        for i in range(j):
            old_diffs.add(rows[r][j] - rows[r][i])
        for m in range(j+1, k+1):
            old_diffs.add(rows[r][m] - rows[r][j])
        
        # Calculate new differences involving index j
        new_diffs = set()
        for i in range(j):
            new_diffs.add(new_val - rows[r][i])
        for m in range(j+1, k+1):
            new_diffs.add(rows[r][m] - new_val)
            
        # Freed differences are those in old but not new
        # Taken differences are those in new but not old
        # The global used set contains all differences from all rows.
        # If we take a difference that is in used but NOT in old_diffs, it's a conflict.
        # Because if it's in old_diffs, it was already "reserved" by this row's old state, 
        # so removing it from the row frees it, and re-taking it is fine.
        
        # Actually, simpler logic:
        # The set of all differences in the system is `used`.
        # When we change row r at index j, the differences `old_diffs` are removed from `used`,
        # and `new_diffs` are added.
        # This is valid if `new_diffs` does not contain any difference that is in `used` 
        # but was NOT in `old_diffs`.
        
        # Let's check:
        for d in new_diffs:
            if d in used and d not in old_diffs:
                return False
        return True

    def apply_move(r, j, new_val):
        # Update used set
        old_diffs = set()
        for i in range(j):
            old_diffs.add(rows[r][j] - rows[r][i])
        for m in range(j+1, k+1):
            old_diffs.add(rows[r][m] - rows[r][j])
            
        for d in old_diffs:
            used.discard(d)
            
        rows[r][j] = new_val
        
        new_diffs = set()
        for i in range(j):
            new_diffs.add(new_val - rows[r][i])
        for m in range(j+1, k+1):
            new_diffs.add(rows[r][m] - new_val)
            
        used.update(new_diffs)

    # Local Search Loop
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Try to find a move that reduces the scope or maintains it
        # We prioritize moves that reduce the maximum element
        
        # Identify the current max
        current_scope = max(max(r) for r in rows)
        
        # Collect all possible moves that are valid
        # To be efficient, we can just iterate through rows and positions
        # and try to decrease the value.
        
        # Strategy: Pick a random row and position, try to decrease.
        # If we find a move that reduces the global max, take it.
        # If we find a move that doesn't increase the global max, take it with some probability.
        
        # Let's try to specifically target the largest elements.
        # Find the row and index with the largest value
        max_val = -1
        max_r = -1
        max_j = -1
        for r in range(n_rows):
            for j in range(1, k+1):
                if rows[r][j] > max_val:
                    max_val = rows[r][j]
                    max_r = r
                    max_j = j
        
        # Try to reduce rows[max_r][max_j]
        moved = False
        # Try values from max_val - 1 down to rows[max_r][max_j-1] + 1
        # To avoid O(N^2) per step, we can just try a few random smaller values
        # or scan downwards if the range is small.
        
        # Since we want to minimize, let's scan downwards from max_val - 1
        # But if the gap is large, this is slow.
        # Let's try a random subset of smaller values first, then scan if needed.
        
        # For now, let's try to find ANY valid move that reduces the value at max_r, max_j
        # If we can't, try other positions.
        
        # Let's iterate through all positions in a random order to avoid bias
        positions = [(r, j) for r in range(n_rows) for j in range(1, k+1)]
        random.shuffle(positions)
        
        for r, j in positions:
            current_val = rows[r][j]
            # Try to decrease
            # If this is the position of the global max, we are very motivated to decrease it.
            # We try a few candidate values.
            
            # Determine a range to search
            # If it's the max position, search more aggressively.
            search_depth = 10 if (r == max_r and j == max_j) else 2
            
            # Try random values in the valid range
            # Range: (rows[r][j-1], current_val)
            low = rows[r][j-1] + 1
            high = current_val - 1
            
            if low > high:
                continue
                
            # Try a few random values
            tried = set()
            for _ in range(search_depth):
                if low >= high:
                    break
                cand = random.randint(low, high)
                if cand in tried:
                    continue
                tried.add(cand)
                
                if can_move(r, j, cand):
                    # Check if this move improves the scope
                    old_scope = current_scope
                    # Simulate scope change
                    # The new max will be max(old_scope excluding current_val if it was the unique max, cand, other maxes)
                    # Simpler: just apply and check.
                    apply_move(r, j, cand)
                    new_scope = max(max(row) for row in rows)
                    
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [row[:] for row in rows]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        # Accept with some probability to escape local minima?
                        # For now, let's accept if it doesn't worsen the scope too much?
                        # Actually, if it doesn't improve best, we should probably revert
                        # unless we are doing simulated annealing.
                        # Let's revert if it doesn't improve the current best, 
                        # but keep the move if it's equal? No, let's stick to hill climbing on current state.
                        # If new_scope == old_scope, it's a sideways move.
                        # If new_scope > old_scope, it's a worsening move.
                        
                        # Let's accept sideways moves to explore.
                        if new_scope <= old_scope:
                            moved = True
                            break
                        else:
                            # Revert
                            # We need to revert the move.
                            # Easiest is to re-apply the old value.
                            # But we modified `used` and `rows`.
                            # Let's just undo by setting back and recalculating used?
                            # That's expensive.
                            # Better: don't apply if we know it's bad?
                            # But we don't know scope until we apply.
                            # Let's just revert manually.
                            pass # This logic is messy. Let's simplify.
            if moved:
                break
        
        if not moved:
            # No move found in this iteration
            # Try a random restart or kick?
            # Given time constraints, maybe just break if we've been stuck for a while?
            # Or try a different strategy:
            # Try to swap two elements?
            # For now, let's just break if no improvement for a long time.
            if no_improve_count > max_no_improve:
                break
            no_improve_count += 1

    return best_rows
