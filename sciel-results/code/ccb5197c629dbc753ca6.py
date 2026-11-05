import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0,
    such that all pairwise differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Lower bound on max entry
    lower_bound = total_diffs  # counting bound
    
    best_result = None
    best_max = float('inf')
    
    # Try different random seeds and strategies
    rng = random.Random(42)
    
    def check_valid(rows):
        """Check if all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def greedy_construct(seed_offset=0):
        """Greedy construction with some randomization."""
        r = random.Random(seed_offset)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def backtracking_construct(max_entry_limit, time_budget=5.0):
        """
        Backtracking to find a valid configuration with max entry <= max_entry_limit.
        Returns (rows, success) or (None, False).
        """
        bt_start = time.time()
        used = set()
        rows = []
        
        def backtrack(row_idx, pos_idx, current_row, last_val):
            if time.time() - bt_start > time_budget:
                return False
            
            if row_idx == n_rows:
                # All rows complete
                if current_row:
                    rows.append(list(current_row))
                return True
            
            if pos_idx == k + 1:
                # Current row complete, move to next row
                rows.append(list(current_row))
                return backtrack(row_idx + 1, 0, [0], 0)
            
            # Try to place the next element
            start_val = last_val + 1
            if pos_idx == 0:
                start_val = 0
            
            for val in range(start_val, max_entry_limit + 1):
                if pos_idx == 0 and val != 0:
                    continue
                
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = val - a
                    if d <= 0:
                        continue
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Forward checking: can we still complete this row?
                # Remaining positions need at least remaining * (remaining+1)/2 more diffs
                remaining = k + 1 - pos_idx - 1
                if remaining > 0:
                    # Check if there's enough room
                    if val + remaining > max_entry_limit:
                        continue
                
                current_row.append(val)
                used.update(new_diffs)
                
                if backtrack(row_idx, pos_idx + 1, current_row, val):
                    return True
                
                current_row.pop()
                used.difference_update(new_diffs)
            
            return False
        
        # Start with first row
        if n_rows > 0:
            success = backtrack(0, 0, [0], 0)
            if success:
                return rows, True
        return None, False
    
    def local_search_improve(rows, max_iterations=50000):
        """
        Try to reduce the maximum entry by local search.
        Swap elements within rows to find better configurations.
        """
        current_rows = [list(r) for r in rows]
        current_max = max(max(r) for r in current_rows)
        best_rows = [list(r) for r in current_rows]
        best_max = current_max
        
        used = set()
        for row in current_rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        iterations = 0
        no_improve_count = 0
        max_no_improve = 5000
        
        while iterations < max_iterations and no_improve_count < max_no_improve:
            iterations += 1
            
            # Pick a random row and position to modify
            row_idx = random.randrange(n_rows)
            row = current_rows[row_idx]
            
            # Pick a position to change (not the first element)
            pos = random.randrange(1, len(row))
            old_val = row[pos]
            
            # Try a new value
            new_val = old_val + random.choice([-1, 1]) * random.randint(1, 3)
            if new_val <= row[pos-1] or new_val >= (row[pos+1] if pos+1 < len(row) else float('inf')):
                continue
            if new_val < 0:
                continue
            
            # Remove old differences
            old_diffs = set()
            for i in range(len(row)):
                if i == pos:
                    continue
                d = abs(row[i] - old_val)
                if d > 0:
                    old_diffs.add(d)
                    used.discard(d)
            
            # Check if new value creates conflicts
            new_diffs = set()
            conflict = False
            for i in range(len(row)):
                if i == pos:
                    continue
                d = abs(row[i] - new_val)
                if d == 0:
                    conflict = True
                    break
                if d in used:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                # Restore old differences
                used.update(old_diffs)
                no_improve_count += 1
                continue
            
            # Accept the move
            row[pos] = new_val
            used.update(new_diffs)
            
            new_max = max(max(r) for r in current_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return best_rows, best_max
    
    # Phase 1: Get an initial solution with greedy
    initial_rows = greedy_construct(0)
    if initial_rows and check_valid(initial_rows):
        best_result = initial_rows
        best_max = max(max(r) for r in initial_rows)
    
    # Phase 2: Try to improve with local search
    if best_result and time.time() < deadline - 30:
        improved_rows, improved_max = local_search_improve(best_result, max_iterations=100000)
        if improved_max < best_max:
            best_result = improved_rows
            best_max = improved_max
    
    # Phase 3: Try backtracking with tighter bounds
    # Try decreasing max_entry from best_max down to lower_bound
    current_limit = best_max if best_max < float('inf') else lower_bound + 50
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Try to find a solution with max entry = current_limit
        limit = current_limit
        if limit < lower_bound:
            break
        
        rows_found, success = backtracking_construct(limit, time_budget=3.0)
        if success and rows_found:
            row_max = max(max(r) for r in rows_found)
            if row_max < best_max:
                best_max = row_max
                best_result = rows_found
                # Try to go lower
                current_limit = row_max - 1
            else:
                current_limit -= 1
        else:
            # Couldn't find with this limit, try a slightly higher one or break
            if limit <= lower_bound + 5:
                break
            current_limit = max(lower_bound, limit - 2)
    
    # Phase 4: Final local search polish
    if best_result and time.time() < deadline - 5:
        improved_rows, improved_max = local_search_improve(best_result, max_iterations=50000)
        if improved_max < best_max:
            best_result = improved_rows
            best_max = improved_max
    
    if best_result is None:
        # Fallback: use greedy
        best_result = greedy_construct(0)
    
    return best_result
