import time
import sys
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # Leave margin for 120s limit
    
    # Lower bound estimate
    # Counting bound: n_rows * k * (k+1) / 2 diffs? No, total diffs is n_rows * k*(k+1)/2.
    # Max diff is M. Diffs are in 1..M.
    # n_rows * k*(k+1)/2 <= M
    # For n=7, k=6: 7 * 21 = 147 diffs.
    # M >= 147? No, diffs don't have to be 1..M contiguous, just distinct.
    # But M must be at least the max diff in any row.
    # The simple counting bound mentioned in the prompt is 105.
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_extension(row, v, used_diffs):
        # Check if adding v to row keeps it valid and disjoint from used_diffs
        # v must be > row[-1]
        if v <= row[-1]:
            return False
        # Check internal uniqueness with existing row elements
        # And global uniqueness
        new_diffs = set()
        for a in row:
            d = v - a
            if d in used_diffs:
                return False
            if d in new_diffs:
                return False
            new_diffs.add(d)
        # Also need to check if any existing diffs in row conflict with new_diffs?
        # No, existing diffs in row are already unique and in used_diffs (if row was added) 
        # or not in used_diffs (if row is partial).
        # Wait, `used_diffs` contains diffs from *completed* rows.
        # The current row's diffs are not in `used_diffs` yet.
        # So we must check if `new_diffs` intersects with `used_diffs` (done above).
        # We must also check if `new_diffs` intersects with *existing diffs in the current row*.
        # My loop `for a in row` generates diffs between v and all a in row.
        # It does NOT check if two *existing* elements in row have a diff that equals a *new* diff?
        # No, the diff is `v - a`. The existing diffs are `b - a` for `b, a` in row.
        # We need to ensure all diffs in the *entire* new row are unique.
        # The new row will have old diffs + new_diffs.
        # Old diffs are unique by construction of the recursion.
        # We need `new_diffs` to be disjoint from `old_diffs` AND `used_diffs`.
        
        # Let's re-calculate old_diffs for efficiency? 
        # Better: pass `current_row_diffs` as a set in the recursion.
        
        return True

    # Revised Backtracking State
    # We track:
    # 1. rows: list of completed rows
    # 2. current_row: list of elements in the row being built
    # 3. current_row_diffs: set of diffs in current_row
    # 4. used_diffs: set of diffs in all completed rows
    
    def backtrack(row_idx, current_row, current_row_diffs, used_diffs, max_val_bound):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
            
        if len(current_row) == k + 1:
            # Row complete
            # Add its diffs to used_diffs
            new_used = used_diffs | current_row_diffs
            
            if row_idx == n_rows - 1:
                # All rows complete
                return True, rows + [current_row]
            
            # Recurse to next row
            # We start the next row with [0]
            # Note: The problem says "starting at 0".
            # So the next row starts with 0.
            # Diffs for [0] is empty.
            success, res = backtrack(row_idx + 1, [0], set(), new_used, max_val_bound)
            if success:
                return True, res
            return False, None
        
        # Determine next value v
        min_v = current_row[-1] + 1
        # We want to try small values first to keep max low
        # But we are bounded by max_val_bound
        
        # Optimization: 
        # If we are in the last row, we can try to be more aggressive?
        
        for v in range(min_v, max_val_bound + 1):
            # Check validity of v
            valid = True
            new_diffs_to_add = set()
            
            for a in current_row:
                d = v - a
                if d in used_diffs:
                    valid = False
                    break
                if d in current_row_diffs:
                    valid = False
                    break
                if d in new_diffs_to_add:
                    valid = False
                    break
                new_diffs_to_add.add(d)
            
            if not valid:
                continue
            
            # Tentatively add v
            new_row = current_row + [v]
            new_current_diffs = current_row_diffs | new_diffs_to_add
            
            success, res = backtrack(row_idx, new_row, new_current_diffs, used_diffs, max_val_bound)
            if success:
                return True, res
            
            # Backtrack
            # (No state to restore as we passed copies, but we can optimize by using mutable state if needed. 
            # For n=7, k=6, copies are cheap enough.)
            
        return False, None

    # We will try increasing max_val_bound
    # Start from a reasonable lower bound
    # The counting bound is 105.
    # Let's start at 105.
    
    # To handle cases where the deterministic search is slow, we can add randomization.
    # But first, let's try deterministic.
    
    # If deterministic fails or times out, we might need to restart with a different strategy.
    # For now, let's just increment the bound.
    
    # Initial bound
    # A safe upper bound is around 150.
    # Let's start at 105.
    
    current_bound = 105
    max_bound = 150
    
    # We might want to randomize the order of `v` slightly if we get stuck?
    # For now, pure "smallest first" is best for minimization.
    
    while current_bound <= max_bound:
        try:
            # Reset state
            rows = []
            # Start first row with [0]
            success, result = backtrack(0, [0], set(), set(), current_bound)
            if success:
                return result
        except TimeoutError:
            # If timeout, break and try to return a greedy solution or increase bound?
            # If we timeout, we haven't found a solution for this bound.
            # We should probably stop and return a best-effort greedy solution.
            break
        
        current_bound += 1
    
    # Fallback: Greedy construction to ensure we return *something* valid
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                found = False
                # Try values starting from row[-1] + 1
                # Limit search to a reasonable range to avoid infinite loop
                for x in range(row[-1] + 1, 200):
                    new_diffs = {x - a for a in row}
                    # Check internal uniqueness
                    if len(new_diffs) != len(row):
                        continue
                    # Check global uniqueness
                    if new_diffs & used:
                        continue
                    # Valid
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    # Expand range
                    for x in range(row[-1] + 1, 500):
                        new_diffs = {x - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                    if not found:
                        # Fallback: pick large random
                        x = row[-1] + 100
                        while True:
                            new_diffs = {x - a for a in row}
                            if len(new_diffs) == len(row) and not (new_diffs & used):
                                row.append(x)
                                used.update(new_diffs)
                                break
                            x += 1
            rows.append(row)
        return rows

    return greedy_init()
