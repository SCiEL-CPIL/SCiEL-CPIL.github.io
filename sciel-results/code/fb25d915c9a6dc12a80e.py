import time
import random
import math
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on max entry
    # We need total_diffs distinct positive differences, each <= max_entry
    # So max_entry >= total_diffs
    lower_bound = total_diffs

    def time_left():
        return deadline - time.time()

    def validate(rows):
        """Check if all differences are distinct. Returns (is_valid, max_entry)."""
        used = set()
        max_entry = 0
        for row in rows:
            if not row:
                continue
            m = max(row)
            if m > max_entry:
                max_entry = m
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False, max_entry
                    used.add(d)
        return True, max_entry

    def get_diffs(row):
        """Get all differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_row(used_diffs, k):
        """
        Greedily construct a row of k+1 increasing integers starting at 0
        that avoids all differences in used_diffs.
        Returns the row, or None if fails within a reasonable bound.
        """
        row = [0]
        x = 0
        max_attempts = 500  # Safety bound
        
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
            else:
                # Try a few more values to see if we can continue
                if max_attempts > 0:
                    max_attempts -= 1
                else:
                    return None
        
        return row

    def greedy_all_rows(n_rows, k):
        """
        Construct all rows greedily. Returns list of rows or None if fails.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used_diffs, k)
            if row is None:
                return None
            rows.append(row)
        return rows

    def backtrack_row(used_diffs, k, max_val):
        """
        Backtrack to find a row of k+1 increasing integers starting at 0
        that avoids all differences in used_diffs, with all values <= max_val.
        Returns the row, or None if no solution exists.
        """
        row = [0]
        
        def backtrack(pos, current_max):
            if pos == k + 1:
                return True
            
            # Try values from current_max+1 to max_val
            for v in range(current_max + 1, max_val + 1):
                new_diffs = {v - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(v)
                    if backtrack(pos + 1, v):
                        return True
                    row.pop()
            
            return False
        
        if backtrack(0, 0):
            return row[:]
        return None

    def local_search(rows, max_val, iterations):
        """
        Local search to minimize max entry.
        Returns (rows, max_entry) after optimization.
        """
        best_rows = [row[:] for row in rows]
        best_max = max(max(row) for row in best_rows)
        
        for _ in range(iterations):
            if time_left() < 0.5:
                break
                
            # Pick a random row and a random element
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)  # Don't change the first element (0)
            
            # Compute valid range for this element
            min_val = best_rows[r_idx][c_idx - 1] + 1
            max_allowed = max_val
            
            if c_idx < k:
                max_allowed = min(max_allowed, best_rows[r_idx][c_idx + 1] - 1)
            
            if min_val > max_allowed:
                continue
            
            # Try a few random values
            for _ in range(3):
                new_val = random.randint(min_val, max_allowed)
                if new_val == best_rows[r_idx][c_idx]:
                    continue
                    
                old_val = best_rows[r_idx][c_idx]
                best_rows[r_idx][c_idx] = new_val
                
                is_valid, curr_max = validate(best_rows)
                
                if is_valid and curr_max < best_max:
                    best_max = curr_max
                else:
                    # Revert
                    best_rows[r_idx][c_idx] = old_val
        
        return best_rows, best_max

    # Strategy:
    # 1. Start with greedy to get a baseline.
    # 2. Try to tighten the max entry using backtracking and local search.
    
    # Initial greedy solution
    baseline_rows = greedy_all_rows(n_rows, k)
    if baseline_rows is None:
        # Fallback: pure greedy without backtracking
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        baseline_rows = rows
    
    best_rows = [row[:] for row in baseline_rows]
    best_max = max(max(row) for row in best_rows)
    
    # Try to improve by reducing the max entry
    # We'll try various max_val values, starting from the current best and going down
    # to the lower bound.
    
    # First, let's try local search on the current solution
    best_rows, best_max = local_search(best_rows, best_max, 2000)
    
    # Now try to find a solution with a smaller max entry using backtracking
    # We'll try max_val values from best_max - 1 down to lower_bound
    
    target_max = best_max - 1
    
    while target_max >= lower_bound and time_left() > 1.0:
        # Try to find a solution with max entry <= target_max
        # We'll use a randomized approach:
        # 1. Try greedy first
        # 2. If that fails, try backtracking with randomization
        
        success = False
        
        # Try a few randomizations
        for attempt in range(10):
            if time_left() < 1.0:
                break
            
            used_diffs = set()
            rows = []
            valid = True
            
            for r in range(n_rows):
                # Try to build a row
                row = [0]
                x = 0
                failed = False
                
                for pos in range(1, k + 1):
                    # Try values up to target_max
                    found = False
                    for v in range(x + 1, target_max + 1):
                        new_diffs = {v - a for a in row}
                        if not new_diffs & used_diffs:
                            row.append(v)
                            used_diffs |= new_diffs
                            x = v
                            found = True
                            break
                    
                    if not found:
                        failed = True
                        break
                
                if failed:
                    valid = False
                    break
                
                rows.append(row)
            
            if valid:
                # Check if this is better than our current best
                curr_max = max(max(row) for row in rows)
                if curr_max < best_max:
                    best_rows = [row[:] for row in rows]
                    best_max = curr_max
                    success = True
                    break
        
        if success:
            # We found a better solution, try to go even lower
            target_max = best_max - 1
        else:
            # This target_max didn't work, try the next one down
            target_max -= 1
    
    # Final local search on the best solution found
    if time_left() > 1.0:
        best_rows, best_max = local_search(best_rows, best_max, 3000)
    
    # Verify the final solution is valid
    is_valid, final_max = validate(best_rows)
    if not is_valid:
        # Fallback to a guaranteed valid solution
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows
    
    return best_rows
