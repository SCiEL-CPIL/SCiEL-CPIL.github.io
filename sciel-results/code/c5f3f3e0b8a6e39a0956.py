import random
import math
import time
import heapq

def construct(n_rows, k):
    row_len = k + 1
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid_extension(row, new_val, used_diffs):
        if new_val <= row[-1]:
            return False
        new_diffs = {new_val - v for v in row}
        if new_diffs & used_diffs:
            return False
        return True

    def greedy_solution():
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for step in range(n_rows * k):
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
                if is_valid_extension(current_row, x, used):
                    current_row.append(x)
                    for v in current_row[:-1]:
                        used.add(x - v)
                    break
                x += 1
        return rows, max(max(r) for r in rows)

    def local_search(rows, used_diffs, time_limit):
        start_time = time.time()
        current_max = max(max(r) for r in rows)
        
        # Helper to check if a specific value can replace rows[r][j]
        def try_decrease(r, j, current_val):
            # We want to find a new_val < current_val
            # Constraints: rows[r][j-1] < new_val < rows[r][j+1] (if j < row_len-1)
            # And new differences must not be in used_diffs (excluding those involving rows[r][j])
            
            lower_bound = rows[r][j-1] + 1
            upper_bound = current_val - 1
            
            if j < row_len - 1:
                upper_bound = min(upper_bound, rows[r][j+1] - 1)
            
            if lower_bound > upper_bound:
                return False
            
            # Collect differences that will be removed
            old_diffs = set()
            for i in range(row_len):
                if i != j:
                    if i < j:
                        old_diffs.add(current_val - rows[r][i])
                    else:
                        old_diffs.add(rows[r][i] - current_val)
            
            # Differences that remain in the row (not involving j)
            # We need to check new diffs against:
            # 1. Diffs from other rows
            # 2. Diffs from same row not involving j
            
            # Optimization: We can just check against the global used_diffs set,
            # but we must temporarily remove old_diffs or check carefully.
            
            # Let's iterate candidate values downwards to find the smallest valid one
            # which minimizes the max.
            
            for new_val in range(upper_bound, lower_bound - 1, -1):
                # Check bounds
                if new_val <= rows[r][j-1]:
                    continue
                if j < row_len - 1 and new_val >= rows[r][j+1]:
                    continue
                    
                # Calculate new diffs
                new_diffs = set()
                for i in range(row_len):
                    if i != j:
                        if i < j:
                            new_diffs.add(new_val - rows[r][i])
                        else:
                            new_diffs.add(rows[r][i] - new_val)
                
                # Check collision
                # A collision occurs if any d in new_diffs is in used_diffs
                # AND d is not in old_diffs (because those are being removed)
                # Actually, if d is in old_diffs, it might still be in used_diffs from ANOTHER row.
                # So we must check:
                # 1. Is d in used_diffs?
                # 2. If yes, is it ONLY from this row's old configuration?
                #    If it's from another row, it's a collision.
                #    If it's from this row's old configuration, it will be removed, so it's fine.
                
                # Efficient check:
                # Let S_other = used_diffs - old_diffs (conceptually)
                # But used_diffs contains diffs from ALL rows.
                # old_diffs are a subset of used_diffs.
                # If a new_diff d is in used_diffs, we need to know if it appears in old_diffs.
                # If d in old_diffs, it means this specific difference value was generated by the old position.
                # If we remove the old position, that instance of d is gone.
                # But if d is also generated by another row, it's still in used_diffs.
                
                # So, we need to count occurrences? Or just check if d is in used_diffs.
                # If d in used_diffs:
                #   If d in old_diffs:
                #      It might be safe if it's ONLY from old_diffs.
                #      But if another row also has d, it's unsafe.
                #   If d not in old_diffs:
                #      It's definitely unsafe (collision with existing).
                
                # To be safe and simple:
                # 1. Remove old_diffs from a copy of used_diffs? No, too slow.
                # 2. Check if any d in new_diffs is in (used_diffs - old_diffs).
                
                # Let's pre-compute "external" diffs? No, used_diffs is global.
                
                # Simple check:
                # For each d in new_diffs:
                #   if d in used_diffs:
                #      # Check if d is in old_diffs
                #      if d in old_diffs:
                #         # It's possible it's from this row only.
                #         # We need to verify if it's from another row.
                #         # This requires knowing which rows generated which diffs.
                #         # This is complex.
                
                # Alternative:
                # Just remove old_diffs from used_diffs, add new_diffs, check for duplicates in the set?
                # No, used_diffs is a set. If a diff is duplicated, it was already invalid.
                # So used_diffs contains unique values.
                # If I remove old_diffs, and add new_diffs, and the size of the set increases by len(new_diffs) - len(old_diffs) + (collisions)?
                # No.
                
                # Let's use a counter or just check against used_diffs carefully.
                # Since the number of diffs per row is small (k=8 -> 36 diffs),
                # and old_diffs has size k (differences with j),
                # we can do:
                
                valid = True
                for d in new_diffs:
                    if d in used_diffs:
                        # It exists in the global set.
                        # Is it ONLY from this row's old state?
                        # If d in old_diffs, it was generated by the old state.
                        # If it was generated by ANOTHER row, it's a collision.
                        # We don't have info on which row generated which diff in `used_diffs`.
                        
                        # Heuristic: If d in old_diffs, we assume it MIGHT be safe, but we need to be sure.
                        # Actually, if d in old_diffs, and we change the value, the old d is gone.
                        # If no other row has d, it's safe.
                        # If another row has d, it's unsafe.
                        
                        # To be strictly correct, we should maintain a count of diffs or a mapping.
                        # For now, let's be conservative:
                        # If d in used_diffs, and d NOT in old_diffs -> Collision.
                        # If d in used_diffs, and d IN old_diffs -> Potentially Safe, but risky.
                        # Let's check if d is in old_diffs. If yes, we allow it, but we must ensure
                        # that when we update used_diffs, we don't create a duplicate.
                        
                        # Actually, if d is in old_diffs, it means the value d was present.
                        # If we remove it, and add it back (if new_diffs also contains d), it's fine.
                        # If new_diffs contains d, and it was in old_diffs, and no other row has it, it's fine.
                        # If another row has it, it's a collision.
                        
                        # Given the rarity of collisions for specific values,
                        # let's just check:
                        # if d in used_diffs:
                        #    if d not in old_diffs:
                        #       valid = False
                        #       break
                        #    # else: d is in old_diffs. It might be shared.
                        #    # We'll tentatively accept, but we must update used_diffs correctly.
                        #    # If we remove old_diffs and add new_diffs, and d is in both,
                        #    # and no other row has d, the count goes 1 -> 0 -> 1. OK.
                        #    # If another row has d, count goes 2 -> 1 -> 2. OK.
                        #    # Wait, if another row has d, then d is in used_diffs.
                        #    # If d is in old_diffs, it's in used_diffs.
                        #    # If we remove old_diffs, d might still be in used_diffs (from other row).
                        #    # If we add new_diffs, and d is in new_diffs, it's already in used_diffs.
                        #    # So no new collision.
                        #    # So if d in old_diffs, it is SAFE to reuse that value, 
                        #    # regardless of whether other rows have it, because it's already "occupied" 
                        #    # and we are just "moving" the source? 
                        #    # No, if another row has d, and this row had d, and we keep d, 
                        #    # it's still a collision (2 rows have d). 
                        #    # BUT, the solution was VALID before, so no two rows had the SAME diff d.
                        #    # Therefore, if d is in used_diffs, it appears EXACTLY ONCE.
                        #    # So if d in old_diffs, it means THIS row was the one that had d.
                        #    # No other row has d.
                        #    # So if we keep d in new_diffs, it's safe!
                        
                        # Conclusion: If d in used_diffs, it must be from this row (since solution is valid).
                        # So if d in old_diffs, it's safe to have d in new_diffs.
                        # If d NOT in old_diffs, it's from another row, so Collision.
                        
                        if d not in old_diffs:
                            valid = False
                            break
                            
                if valid:
                    # Update rows and used_diffs
                    # Remove old_diffs
                    for d in old_diffs:
                        used_diffs.discard(d)
                    # Add new_diffs
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    # Update row
                    rows[r][j] = new_val
                    
                    # Recalculate max
                    # We can just return True and let the caller check max
                    return True
            return False

        while time.time() - start_time < time_limit:
            # Random move
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, row_len - 1)
            
            if try_decrease(r, j, rows[r][j]):
                new_max = max(max(row) for row in rows)
                if new_max < current_max:
                    current_max = new_max
                    # Optional: Log or track best
            else:
                # Try to increase? No, we want to minimize max.
                # Maybe try a different row?
                pass
                
        return rows, current_max

    start_time = time.time()
    timeout = 115
    
    # Get initial solution
    rows, cost = greedy_solution()
    used = get_used_diffs(rows)
    
    # Run local search
    rows, cost = local_search(rows, used, timeout)
    
    return rows
