"""Rewritten with randomized greedy + local search for difference triangle sets."""
import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s kill
    
    best_rows = None
    best_max = float('inf')
    
    # Helper: compute max entry of a solution
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    # Helper: check validity (all positive differences distinct across all rows)
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    # Greedy row construction with randomization
    def greedy_row(used, k, randomize=False, bias=0.0):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                if randomize and bias > 0 and random.random() < bias:
                    # Sometimes skip a valid option to explore alternatives
                    continue
                row.append(x)
                used |= new
        return row
    
    # Local search: try to reduce max entry by swapping values
    def local_search(rows, used_diffs, max_entry_limit):
        """Try to improve by reducing the maximum entry."""
        if not is_valid(rows):
            return rows, used_diffs
        
        current_max = max_entry(rows)
        if current_max >= max_entry_limit:
            return rows, used_diffs
            
        # Try to find a row where we can reduce the last element
        for attempt in range(50):
            row_idx = random.randint(0, len(rows) - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue
            
            # Try reducing the last element
            old_last = row[-1]
            if old_last <= 1:
                continue
            
            # Find the largest element we can reduce to
            new_last = old_last - 1
            while new_last > 0:
                # Check if reducing last element keeps validity
                test_row = row[:-1] + [new_last]
                if test_row == row:
                    break
                # Must be strictly increasing
                if new_last <= test_row[-2]:
                    break
                    
                # Check differences
                new_diffs = set()
                for i in range(len(test_row)):
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        new_diffs.add(d)
                        
                # Remove old differences, add new ones
                old_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        old_diffs.add(row[j] - row[i])
                        
                # Check if new diffs conflict with other rows' diffs
                other_diffs = set()
                for ri, r in enumerate(rows):
                    if ri == row_idx:
                        continue
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            other_diffs.add(r[j] - r[i])
                            
                if not (new_diffs & other_diffs):
                    # Valid reduction
                    rows[row_idx] = test_row
                    # Update used_diffs
                    used_diffs -= old_diffs
                    used_diffs |= new_diffs
                    break
                new_last -= 1
        
        return rows, used_diffs
    
    # Multi-start randomized greedy
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 12345)  # Deterministic but varied
        
        used = set()
        rows = []
        success = True
        
        for r in range(n_rows):
            if time.time() > deadline:
                success = False
                break
            
            # Use different randomization strategies
            bias = 0.1 + 0.3 * random.random()
            row = greedy_row(used, k, randomize=True, bias=bias)
            
            if not row:
                success = False
                break
                
            rows.append(row)
        
        if not success or len(rows) < n_rows:
            continue
        
        # Verify validity
        if not is_valid(rows):
            continue
        
        current_max = max_entry(rows)
        
        if current_max < best_max:
            best_max = current_max
            best_rows = [list(row) for row in rows]
            
            # Apply local search to try to improve further
            used_diffs = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used_diffs.add(row[j] - row[i])
                        
            rows, used_diffs = local_search(rows, used_diffs, best_max)
            
            if is_valid(rows):
                improved_max = max_entry(rows)
                if improved_max < best_max:
                    best_max = improved_max
                    best_rows = [list(row) for row in rows]
    
    # If no valid solution found, fall back to simple greedy
    if best_rows is None:
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
# EVOLVE-BLOCK-END
