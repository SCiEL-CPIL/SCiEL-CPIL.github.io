import random
import math
import time

def construct(n_rows, k):
    row_len = k + 1
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid_val(row, idx, new_val, used_diffs_excl_row):
        # Check if new_val at row[idx] is valid
        # 1. Increasing order
        if idx > 0 and new_val <= row[idx-1]:
            return False
        if idx < len(row) - 1 and new_val >= row[idx+1]:
            return False
            
        # 2. Difference uniqueness
        # Differences involving new_val: new_val - row[i] for i < idx
        # and row[j] - new_val for j > idx
        
        # We need to check against:
        # a) Other differences in the same row (excluding those involving idx)
        # b) Differences in other rows (used_diffs_excl_row)
        
        row_diffs = set()
        for i in range(len(row)):
            if i == idx:
                continue
            if i < idx:
                row_diffs.add(new_val - row[i])
            else:
                row_diffs.add(row[i] - new_val)
        
        # Check for duplicates within the same row's new diffs
        if len(row_diffs) != len([d for d in row_diffs]): # This check is flawed, set already handles uniqueness, but we need to check against OTHER diffs in the row not involving idx?
             # Actually, if we change row[idx], the diffs involving idx change.
             # The other diffs in the row (not involving idx) remain the same.
             # We must ensure the NEW diffs do not collide with:
             # 1. The OLD diffs in the same row that do NOT involve idx.
             # 2. The diffs in other rows.
             
             pass
             
        # Let's do it properly:
        # 1. Calculate set of diffs in this row that do NOT involve idx.
        stable_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                if i != idx and j != idx:
                    stable_diffs.add(row[j] - row[i])
                    
        # 2. Calculate new diffs involving idx
        new_diffs = set()
        for i in range(len(row)):
            if i == idx:
                continue
            if i < idx:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)
                
        # 3. Check collisions
        # Collision within row: new_diffs & stable_diffs
        if new_diffs & stable_diffs:
            return False
        # Collision with other rows
        if new_diffs & used_diffs_excl_row:
            return False
            
        return True

    def greedy_solution():
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for step in range(n_rows * k):
            # Pick row with smallest length, tie-break by smallest last val
            min_len = row_len
            target_row_idx = -1
            min_last_val = float('inf')
            for i in range(n_rows):
                if len(rows[i]) < min_len:
                    min_len = len(rows[i])
                    target_row_idx = i
                    min_last_val = rows[i][-1]
                elif len(rows[i]) == min_len:
                    if rows[i][-1] < min_last_val:
                        min_last_val = rows[i][-1]
                        target_row_idx = i
            if target_row_idx == -1:
                break
            i = target_row_idx
            current_row = rows[i]
            x = current_row[-1] + 1
            while True:
                new_diffs = {x - v for v in current_row}
                if not (new_diffs & used):
                    current_row.append(x)
                    used.update(new_diffs)
                    break
                x += 1
        return rows, max(max(r) for r in rows)

    start_time = time.time()
    timeout = 110
    
    # 1. Get initial solution
    rows, current_max = greedy_solution()
    best_rows = [r[:] for r in rows]
    best_max = current_max
    used_diffs = get_used_diffs(rows)
    
    # 2. Local Search: Try to reduce values
    while time.time() - start_time < timeout:
        improved = False
        # Sort rows by max element descending to target the largest ones first
        row_indices = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
        
        for i in row_indices:
            if time.time() - start_time > timeout:
                break
            row = rows[i]
            # Try to reduce elements from last to first
            for j in range(len(row) - 1, 0, -1):
                if time.time() - start_time > timeout:
                    break
                current_val = row[j]
                min_val = row[j-1] + 1
                # Try values from current_val - 1 down to min_val
                # Optimization: Don't try all, maybe step? No, try all for small k.
                for new_val in range(current_val - 1, min_val - 1, -1):
                    # Check validity
                    # 1. Stable diffs in this row (not involving j)
                    stable_diffs = set()
                    for a in range(len(row)):
                        for b in range(a+1, len(row)):
                            if a != j and b != j:
                                stable_diffs.add(row[b] - row[a])
                    
                    # 2. New diffs involving j
                    new_diffs = set()
                    for a in range(len(row)):
                        if a == j:
                            continue
                        if a < j:
                            new_diffs.add(new_val - row[a])
                        else:
                            new_diffs.add(row[a] - new_val)
                            
                    # 3. Diffs in other rows
                    # We need used_diffs excluding this row's current diffs?
                    # Actually, we can just check if new_diffs intersects with:
                    # (a) stable_diffs
                    # (b) all diffs in other rows
                    
                    # Efficient check:
                    # We know used_diffs contains ALL diffs.
                    # We need to ensure new_diffs does not intersect with:
                    # - stable_diffs
                    # - (used_diffs - old_diffs_involving_j)
                    
                    # Let's compute old_diffs_involving_j
                    old_diffs = set()
                    for a in range(len(row)):
                        if a == j:
                            continue
                        if a < j:
                            old_diffs.add(current_val - row[a])
                        else:
                            old_diffs.add(row[a] - current_val)
                            
                    # Diffs in other rows = used_diffs - old_diffs (since old_diffs are part of used_diffs)
                    # But wait, stable_diffs are also in used_diffs.
                    # So, forbidden = (used_diffs - old_diffs)
                    # We need new_diffs & forbidden == empty
                    # AND new_diffs & stable_diffs == empty?
                    # stable_diffs is a subset of (used_diffs - old_diffs).
                    # So checking new_diffs & (used_diffs - old_diffs) == empty is sufficient.
                    
                    forbidden = used_diffs - old_diffs
                    if not (new_diffs & forbidden):
                        # Valid move
                        # Update row
                        row[j] = new_val
                        # Update used_diffs
                        used_diffs -= old_diffs
                        used_diffs |= new_diffs
                        improved = True
                        # Update current_max if this was the max element
                        if j == len(row) - 1:
                            if new_val < best_max:
                                best_max = new_val
                                best_rows = [r[:] for r in rows]
                        break # Move to next j (lower index) or next row?
                        # Breaking j loop means we restart the reduction for this row?
                        # Or just proceed to next j?
                        # Proceeding to next j (j-1) is good because reducing a later element might allow reducing an earlier one?
                        # No, reducing earlier element doesn't help later one's validity directly in terms of "value", but it might free up differences.
                        # Let's just break to re-evaluate the row from the top?
                        # Actually, just continuing to j-1 is fine.
        
        if not improved:
            # Try random perturbation to escape local minimum
            # Pick a random row and a random element (not 0)
            # Increase it by a random amount, then try to reduce it back
            # Or just random restart?
            # Random restart is expensive.
            # Let's try "kick": pick a random row, increase the last element by 1-5, then run reduction.
            if random.random() < 0.5:
                i = random.randint(0, n_rows - 1)
                j = len(rows[i]) - 1
                # Increase last element
                old_val = rows[i][j]
                # Find a valid increase
                # We need to ensure new diffs don't collide.
                # This is harder.
                # Alternative: Just restart the local search with a slightly different initial solution?
                # Or shuffle the greedy order?
                
                # Let's try to randomize the greedy construction again and see if we get a better start
                # But greedy is deterministic.
                # Let's add randomness to greedy: pick random valid x from a small window?
                # For now, if stuck, we just break or continue waiting.
                # Since we have time, we can try multiple greedy variants.
                pass

    return best_rows
