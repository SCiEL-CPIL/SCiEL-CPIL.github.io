import time
import random
import sys
import math
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Each row has k+1 elements starting at 0.
    # Number of differences per row = (k+1)*k/2
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    # Estimate max possible difference for array sizing.
    # If we fail, we can resize, but 10000 is usually safe for small n, k.
    # The known best for (4,7) is 134. Let's start with a reasonable bound.
    # If the greedy construction exceeds this, we increase the bound.
    max_scope_limit = 2000 

    def get_diffs_array(rows):
        """
        Returns a boolean array of size max_scope_limit+1 where arr[d] is True if diff d is used.
        """
        arr = np.zeros(max_scope_limit + 1, dtype=bool)
        for row in rows:
            # Use numpy to compute diffs for a single row quickly? 
            # Or just loop since k is small.
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= max_scope_limit:
                        arr[d] = True
                    else:
                        # If we exceed limit, we need to expand. 
                        # For now, assume we stay within bounds or handle resize outside.
                        pass
        return arr

    def greedy_construct(scope_limit=None):
        """
        Greedy construction: build rows one by one, each row greedily.
        Uses a set for speed during construction (simplest), then converts.
        """
        if scope_limit is None:
            scope_limit = max_scope_limit
            
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                # Try small increments first
                for x in range(last_val + 1, scope_limit + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def local_search(rows, time_limit):
        """
        Local search using numpy for fast difference checking.
        Moves: 
        1. Shrink: Try to reduce the last element of a row.
        2. Move: Try to move an internal element to a smaller valid position.
        """
        rows = [r[:] for r in rows]
        
        # Determine max scope for array
        current_max = max(max(r) for r in rows)
        arr_size = max(current_max + 100, 500)
        
        # Initialize difference array
        diff_arr = np.zeros(arr_size, dtype=bool)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d < arr_size:
                        diff_arr[d] = True
        
        def update_diffs(row_idx, pos, old_val, new_val, diff_arr):
            """
            Updates diff_arr for a change in row_idx at pos from old_val to new_val.
            Returns None if invalid (conflict), otherwise updates diff_arr in place.
            """
            row = rows[row_idx]
            
            # 1. Collect old diffs and clear them
            old_diffs = []
            for i in range(len(row)):
                if i != pos:
                    if i < pos:
                        d = old_val - row[i]
                    else:
                        d = row[i] - old_val
                    old_diffs.append(d)
            
            # Check if any old diff is still needed? No, we are removing them.
            # But wait, if another part of the row uses the same diff? 
            # In a strictly increasing row, diffs (a_j - a_i) are unique for pairs (i,j)?
            # No, a_j - a_i can be equal to a_l - a_m for different pairs? 
            # The problem statement says: "all differences between two entries of the same row are distinct over all rows".
            # This implies that within a row, all differences must be distinct? 
            # "distinct over all rows" usually means the set of all differences from all rows must be disjoint.
            # Does it imply differences within a single row must be distinct?
            # "Choose 4 rows ... so that all differences between two entries of the same row are distinct over all rows"
            # This phrasing is slightly ambiguous. 
            # Standard Sidon set / Golomb ruler definition: A Golomb ruler is a set of marks such that all differences between pairs of marks are distinct.
            # The task says "rows of 8 increasing integers". 
            # If it's a Golomb ruler per row, then yes, diffs within a row are distinct.
            # "distinct over all rows" means the union of all differences from all rows is a set (no duplicates).
            # So, if a diff appears in Row 1, it cannot appear in Row 2, nor in Row 1 again.
            # Therefore, within a row, all diffs are unique.
            # So we can safely clear the old diffs.
            
            for d in old_diffs:
                if d < arr_size:
                    diff_arr[d] = False
            
            # 2. Check new diffs
            new_diffs = []
            for i in range(len(row)):
                if i != pos:
                    if i < pos:
                        d = new_val - row[i]
                    else:
                        d = row[i] - new_val
                    if d >= arr_size:
                        return None # Too big, invalid
                    if diff_arr[d]:
                        # Conflict!
                        # Restore old diffs
                        for od in old_diffs:
                            if od < arr_size:
                                diff_arr[od] = True
                        return None
                    new_diffs.append(d)
            
            # 3. Commit new diffs
            for d in new_diffs:
                diff_arr[d] = True
            
            # Update row
            row[pos] = new_val
            return True

        best_rows = [r[:] for r in rows]
        best_scope = max(max(r) for r in rows)
        
        t_end = time.time() + time_limit
        
        while time.time() < t_end:
            improved = False
            # Try to shrink the largest element of each row
            for r_idx in range(n_rows):
                if time.time() > t_end:
                    break
                row = rows[r_idx]
                current_max_val = row[-1]
                # Try to reduce the last element
                # We can try to move it to a smaller value, or move other elements to allow it to shrink?
                # Actually, moving the last element to a smaller value is a direct shrink.
                # But we must maintain increasing order.
                # Let's try to move the last element to the smallest possible valid value > row[-2].
                
                # First, clear the diffs involving the last element
                pos = len(row) - 1
                old_val = row[pos]
                
                # Find smallest valid new_val
                min_val = row[pos - 1] + 1
                # We can't just scan from min_val upwards if the range is huge.
                # But we want to minimize scope, so we want the smallest valid val.
                
                # To do this efficiently, we need to know which diffs are "blocked".
                # A value x is valid if (x - row[i]) is not in diff_arr for all i < pos.
                
                # Let's try a few candidates near the current value or lower.
                # Actually, a "shrink" move implies we found a hole.
                # Let's try to replace the last element with the smallest valid integer > row[-2].
                
                # Optimization: 
                # 1. Remove diffs of last element.
                # 2. Scan from row[-2]+1 upwards until we find a valid x.
                # 3. If x < old_val, accept.
                
                # Remove diffs
                old_diffs = []
                for i in range(pos):
                    d = old_val - row[i]
                    old_diffs.append(d)
                    diff_arr[d] = False
                
                # Scan for new value
                found_new = False
                for x in range(min_val, old_val + 1):
                    valid = True
                    for i in range(pos):
                        d = x - row[i]
                        if diff_arr[d]:
                            valid = False
                            break
                    if valid:
                        # Check if x creates new conflicts with itself? No, unique within row.
                        # Check if x is within bounds
                        if x < arr_size:
                            # Commit
                            for i in range(pos):
                                diff_arr[x - row[i]] = True
                            row[pos] = x
                            found_new = True
                            break
                
                if not found_new:
                    # Restore old state
                    for d in old_diffs:
                        diff_arr[d] = True
                else:
                    improved = True
                    current_scope = max(max(r) for r in rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
            
            if not improved:
                # No simple shrink found. Try random local moves to escape local minima.
                # Pick a random row and a random position (except 0).
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, len(rows[r_idx]) - 1)
                old_val = rows[r_idx][pos]
                
                # Try to move to a random valid position in a small window around old_val
                # Or try to move to the smallest valid position (greedy local move)
                
                # Remove diffs
                old_diffs = []
                row = rows[r_idx]
                for i in range(len(row)):
                    if i != pos:
                        if i < pos:
                            d = old_val - row[i]
                        else:
                            d = row[i] - old_val
                        old_diffs.append(d)
                        diff_arr[d] = False
                
                # Try to find a new valid value.
                # We want to keep the scope small, so prefer values close to old_val or smaller.
                # Let's try a few random candidates in [row[pos-1]+1, row[pos+1]-1] (if pos < len-1)
                min_v = row[pos - 1] + 1
                max_v = row[pos + 1] - 1 if pos < len(row) - 1 else best_scope + 10
                
                # Try 10 random candidates
                found_new = False
                for _ in range(10):
                    if min_v > max_v:
                        break
                    x = random.randint(min_v, max_v)
                    if x == old_val:
                        continue
                    valid = True
                    for i in range(len(row)):
                        if i != pos:
                            if i < pos:
                                d = x - row[i]
                            else:
                                d = row[i] - x
                            if d >= arr_size or diff_arr[d]:
                                valid = False
                                break
                    if valid:
                        # Commit
                        for i in range(len(row)):
                            if i != pos:
                                if i < pos:
                                    d = x - row[i]
                                else:
                                    d = row[i] - x
                                diff_arr[d] = True
                        row[pos] = x
                        found_new = True
                        break
                
                if not found_new:
                    # Restore
                    for d in old_diffs:
                        diff_arr[d] = True

        return best_rows, best_scope

    # Step 1: Greedy construction
    best_rows = None
    best_scope = float('inf')

    # Try to construct with a tight scope first
    # The counting bound is 112. Known best is 134.
    # Let's try to construct greedily and see what scope we get.
    
    for restart in range(5):
        if time.time() > deadline - 30:
            break
            
        rows = greedy_construct(scope_limit=500)
        if rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                
    if best_rows is None:
        # Fallback to larger scope
        rows = greedy_construct(scope_limit=5000)
        if rows:
            best_rows = rows
            best_scope = max(max(r) for r in rows)
        else:
            # Emergency fallback: just return something valid, even if large
            used_diffs = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for y in row:
                        if x - y in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                rows.append(row)
            return rows

    # Step 2: Local Search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_rows, best_scope = local_search(best_rows, remaining_time - 5)

    return best_rows
