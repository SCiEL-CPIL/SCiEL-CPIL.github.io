import time
import random
import sys

def construct(n_rows, k):
    """
    Find a set of n_rows rows, each with k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a greedy initialization followed by iterative local search.
    """
    start_time = time.time()
    time_limit = 115  # seconds
    
    def greedy(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def compute_diffs(row):
        """Compute all positive differences in a row."""
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}
    
    def check_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.append(row[j] - row[i])
        return len(all_diffs) == len(set(all_diffs))
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def random_row(used_diffs, max_val, k):
        """
        Try to generate a random valid row with k+1 elements starting at 0,
        all <= max_val, with differences disjoint from used_diffs.
        Returns None if fails.
        """
        # Try multiple random restarts
        for attempt in range(50):
            row = [0]
            success = True
            for _ in range(k):
                # Choose a random next element
                # We need to find x > row[-1] such that {x - a for a in row} is disjoint from used_diffs
                candidates = []
                # Sample some candidates randomly
                for _ in range(200):
                    x = random.randint(row[-1] + 1, max_val)
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append((x, new_diffs))
                
                if not candidates:
                    success = False
                    break
                
                # Pick the smallest candidate (greedy within random)
                candidates.sort()
                x, new_diffs = candidates[0]
                row.append(x)
                used_diffs = used_diffs | new_diffs
            
            if success and len(row) == k + 1:
                return row
        
        return None
    
    def local_search(rows, used_diffs, scope):
        """
        Try to improve the solution by modifying individual rows.
        Returns improved rows and used_diffs, or None if no improvement.
        """
        best_rows = [list(r) for r in rows]
        best_used = set(used_diffs)
        best_scope = scope
        
        for _ in range(2000):  # Limit iterations
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random row to try to improve
            idx = random.randint(0, n_rows - 1)
            old_row = list(best_rows[idx])
            old_diffs = compute_diffs(old_row)
            
            # Remove this row's diffs from used set
            temp_used = best_used - old_diffs
            
            # Try to generate a new row with smaller max
            max_old = old_row[-1]
            new_max_limit = max_old - 1  # Try to reduce the max
            
            if new_max_limit < k:  # Minimum possible max for k+1 elements starting at 0
                continue
            
            new_row = None
            for attempt in range(100):
                if time.time() - start_time > time_limit:
                    break
                # Try to build a random row with max <= new_max_limit
                row = [0]
                success = True
                for _ in range(k):
                    # Try to find a valid next element
                    found = False
                    # Try random values
                    for _ in range(50):
                        x = random.randint(row[-1] + 1, new_max_limit)
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & temp_used):
                            row.append(x)
                            temp_used = temp_used | new_diffs
                            found = True
                            break
                    
                    if not found:
                        success = False
                        break
                
                if success and len(row) == k + 1:
                    new_row = row
                    break
            
            if new_row is not None:
                new_diffs = compute_diffs(new_row)
                new_scope = max(max(r) for r in best_rows[:idx] + [new_row] + best_rows[idx+1:])
                
                if new_scope < best_scope:
                    best_rows[idx] = new_row
                    best_used = (best_used - old_diffs) | new_diffs
                    best_scope = new_scope
        
        return best_rows, best_used, best_scope
    
    # Step 1: Greedy initialization
    rows = greedy(n_rows, k)
    scope = get_scope(rows)
    used_diffs = set()
    for row in rows:
        used_diffs |= compute_diffs(row)
    
    best_rows = [list(r) for r in rows]
    best_scope = scope
    best_used = set(used_diffs)
    
    # Step 2: Iterative local search
    # Try multiple rounds of local search
    for round_num in range(20):
        if time.time() - start_time > time_limit:
            break
        
        rows, used_diffs, scope = local_search(best_rows, best_used, best_scope)
        
        if scope < best_scope:
            best_rows = rows
            best_used = used_diffs
            best_scope = scope
        else:
            # No improvement, try a different random seed or break
            random.seed(round_num * 12345 + 67890)
    
    # Step 3: Final attempt with smaller scope bound
    if time.time() - start_time < time_limit - 5:
        # Try to directly search for a solution with scope < best_scope
        target_scope = best_scope - 1
        
        if target_scope >= k:  # Minimum possible
            # Try to construct all rows with max <= target_scope
            for attempt in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                test_rows = []
                test_used = set()
                success = True
                
                for i in range(n_rows):
                    # Generate a row with max <= target_scope
                    row = [0]
                    row_success = True
                    for _ in range(k):
                        found = False
                        # Try values from target_scope down to row[-1] + 1
                        # or random
                        for x in range(target_scope, row[-1], -1):
                            new_diffs = {x - a for a in row}
                            if not (new_diffs & test_used):
                                row.append(x)
                                test_used |= new_diffs
                                found = True
                                break
                        
                        if not found:
                            row_success = False
                            break
                    
                    if row_success and len(row) == k + 1:
                        test_rows.append(row)
                    else:
                        success = False
                        break
                
                if success:
                    new_scope = max(max(r) for r in test_rows)
                    if new_scope < best_scope:
                        best_rows = test_rows
                        best_used = test_used
                        best_scope = new_scope
                        break
    
    return best_rows
