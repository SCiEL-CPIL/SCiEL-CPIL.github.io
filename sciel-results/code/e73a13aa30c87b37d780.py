# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # m = k + 1
    m = k + 1
    
    def solve_csp(n_rows, k, max_entry_limit, rng):
        """
        Backtracking search to find n_rows of k+1 integers.
        """
        m = k + 1
        used_diffs = [False] * (max_entry_limit + 1)
        rows = []
        
        def check_diffs(row, used_diffs, max_val):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d > max_val:
                        return False
                    if used_diffs[d]:
                        return False
            return True

        def mark_diffs(row, used_diffs, mark=True):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used_diffs[d] = mark

        def backtrack(row_idx):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            current_row = [0]
            
            def build_row(pos):
                if pos == m:
                    if check_diffs(current_row, used_diffs, max_entry_limit):
                        mark_diffs(current_row, used_diffs, True)
                        rows.append(list(current_row))
                        
                        if backtrack(row_idx + 1):
                            return True
                        
                        rows.pop()
                        mark_diffs(current_row, used_diffs, False)
                    return False
                
                start_val = current_row[-1] + 1
                end_val = max_entry_limit
                
                candidates = list(range(start_val, end_val + 1))
                rng.shuffle(candidates)
                
                for val in candidates:
                    current_row.append(val)
                    
                    valid = True
                    for prev_val in current_row[:-1]:
                        d = val - prev_val
                        if d > max_entry_limit or used_diffs[d]:
                            valid = False
                            break
                    
                    if valid and val + (m - pos - 1) <= max_entry_limit:
                        if build_row(pos + 1):
                            return True
                    
                    current_row.pop()
                
                return False
            
            return build_row(1)
        
        return backtrack(0)

    best_rows = None
    best_max = float('inf')
    rng = random.Random(42)
    
    # Try known best first, then slightly larger
    for max_limit in [112, 113, 114, 115, 116, 120, 130, 150]:
        if time.time() > deadline - 5.0:
            break
            
        # Limit attempts to avoid wasting all time on one limit
        # If we don't find a solution in 20s for a limit, move to next?
        # Or just rely on the global deadline.
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
                
            success = solve_csp(n_rows, k, max_limit, rng)
            
            if success:
                # Validate
                used = set()
                valid = True
                for row in rows:
                    if len(row) != k + 1 or row[0] != 0:
                        valid = False
                        break
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in used:
                                valid = False
                                break
                            used.add(d)
                        if not valid: break
                    if not valid: break
                
                if valid:
                    cur_max = max(max(r) for r in rows)
                    if best_rows is None or cur_max < best_max:
                        best_rows = [list(r) for r in rows]
                        best_max = cur_max
                    if best_max == 112:
                        return best_rows
                # If we found a valid solution, we can break the attempt loop
                # for this limit, as we found the best for this limit.
                # But we might find a better one (smaller max) in a later attempt?
                # No, max_limit is fixed for this loop iteration.
                # So the max entry will be <= max_limit.
                # We want the global min.
                # Since we iterate max_limit upwards, the first valid solution 
                # for a given limit is the best we can get for that limit.
                # So we can break the attempt loop.
                break 

    # Fallback: Greedy construction
    if best_rows is None:
        used = [False] * 300
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d >= len(used):
                            used.extend([False] * 100)
                        if used[d]:
                            valid = False
                            break
                    if valid:
                        break
                    x += 1
                row.append(x)
                for a in row[:-1]:
                    used[x - a] = True
            rows.append(row)
        best_rows = rows
        
    return best_rows

# EVOLVE-BLOCK-END
