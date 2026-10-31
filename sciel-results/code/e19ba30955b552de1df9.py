import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = k + 1  # number of entries per row
    num_diffs_per_row = n * (n - 1) // 2  # C(n,2)
    
    def time_left():
        return deadline - time.time()
    
    def is_valid_partial(row, used_diffs):
        """Check if adding differences from row to used_diffs causes conflicts."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs or d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def greedy_build_row(used_diffs, max_val_hint):
        """Build a single row greedily, trying to use small differences."""
        row = [0]
        while len(row) < n:
            found = False
            # Try candidates in order of increasing value
            for x in range(row[-1] + 1, max_val_hint + 1):
                # Check if adding x creates any conflict
                conflict = False
                local_diffs = set()
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        conflict = True
                        break
                    local_diffs.add(d)
                if not conflict:
                    row.append(x)
                    used_diffs |= local_diffs
                    found = True
                    break
            if not found:
                return None
        return row
    
    # Strategy: Try different approaches and keep the best
    best_rows = None
    best_max = float('inf')
    
    # Approach 1: Greedy with increasing max_val hints
    # Start with the counting bound and increase
    lower_bound = n_rows * num_diffs_per_row  # minimum possible max value
    
    # Try greedy construction with various strategies
    for attempt in range(50):
        if time_left() < 5:
            break
        
        if attempt == 0:
            # Simple greedy
            used_diffs = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = greedy_build_row(used_diffs, 200)
                if row is None:
                    success = False
                    break
                rows.append(row)
            if success:
                max_val = max(max(r) for r in rows)
                if max_val < best_max:
                    best_max = max_val
                    best_rows = rows
        else:
            # Backtracking with limited depth
            if time_left() < 10:
                break
            
            def backtrack(rows, used_diffs, row_idx):
                nonlocal best_max, best_rows
                if time_left() < 1:
                    return False
                if row_idx == n_rows:
                    max_val = max(max(r) for r in rows)
                    if max_val < best_max:
                        best_max = max_val
                        best_rows = list(rows)
                    return True
                
                # Try building row with backtracking on last few positions
                row = [0]
                # Greedily fill first n-2 positions
                for pos in range(1, n - 2):
                    found = False
                    for x in range(row[-1] + 1, 150):
                        conflict = False
                        local_diffs = set()
                        for a in row:
                            d = x - a
                            if d in used_diffs or d in local_diffs:
                                conflict = True
                                break
                            local_diffs.add(d)
                        if not conflict:
                            row.append(x)
                            used_diffs |= local_diffs
                            found = True
                            break
                    if not found:
                        return False
                
                # Now try the last 2 positions with backtracking
                def backtrack_last2(row, used_diffs, pos):
                    nonlocal best_max, best_rows
                    if time_left() < 0.5:
                        return False
                    if pos == n:
                        max_val = max(max(r) for r in rows)
                        if max_val < best_max:
                            best_max = max_val
                            best_rows = list(rows)
                        return True
                    
                    last_val = row[-1]
                    for x in range(last_val + 1, 150):
                        conflict = False
                        local_diffs = set()
                        for a in row:
                            d = x - a
                            if d in used_diffs or d in local_diffs:
                                conflict = True
                                break
                            local_diffs.add(d)
                        if not conflict:
                            row.append(x)
                            new_used = used_diffs | local_diffs
                            if backtrack_last2(row, new_used, pos + 1):
                                return True
                            row.pop()
                    return False
                
                if backtrack_last2(row, used_diffs, n - 2):
                    return True
                return False
            
            used_diffs = set()
            rows = []
            backtrack(rows, used_diffs, 0)
    
    # If no better solution found, use the greedy result
    if best_rows is None:
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_build_row(used_diffs, 200)
            if row is None:
                # Fallback: just keep going
                row = [0]
                while len(row) < n:
                    x = row[-1] + 1
                    while True:
                        conflict = False
                        local_diffs = set()
                        for a in row:
                            d = x - a
                            if d in used_diffs or d in local_diffs:
                                conflict = True
                                break
                            local_diffs.add(d)
                        if not conflict:
                            break
                        x += 1
                    row.append(x)
                    used_diffs |= local_diffs
            rows.append(row)
        best_rows = rows
    
    return best_rows
# EVOLVE-BLOCK-END
