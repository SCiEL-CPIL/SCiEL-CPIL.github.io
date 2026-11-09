import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for overhead
    
    # Greedy initial solution
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    
    best_rows = [r[:] for r in rows]
    best_span = max(max(r) for r in best_rows)
    
    # Backtracking search to improve
    # We try to find a solution with span < best_span
    # We will search for span from 112 (counting bound) up to best_span - 1
    # Or simply use the current best_span as the upper bound and try to find any solution,
    # then tighten the bound.
    
    # Actually, it's better to just run a backtracking search with the current best_span
    # as the limit, and if it finds a solution, update best_span and retry with a tighter limit.
    # But backtracking is expensive. Let's try a heuristic local search instead?
    # No, for n=4, k=7, the search space is small enough for backtracking with good pruning.
    
    def backtrack(row_idx, current_rows, used_diffs, max_span_limit):
        nonlocal best_rows, best_span
        
        if time.time() - start_time > time_limit:
            return False
        
        if row_idx == n_rows:
            # Check if we improved
            current_max = max(max(r) for r in current_rows)
            if current_max < best_span:
                best_span = current_max
                best_rows = [r[:] for r in current_rows]
                return True
            return False
        
        # Build the next row
        row = [0]
        # We need to choose k more elements
        # The elements must be increasing
        # The differences with existing elements in the row must not be in used_diffs
        # The differences within the new row must also be distinct from each other and used_diffs
        
        # We can use a recursive helper to build the row
        def build_row(pos, current_row, local_diffs):
            nonlocal used_diffs
            
            if time.time() - start_time > time_limit:
                return None
            
            if pos == k + 1:
                # Row complete
                # Check if max element is within limit
                if current_row[-1] > max_span_limit:
                    return None
                
                # Update global state
                new_used = used_diffs | local_diffs
                current_rows.append(current_row)
                
                if backtrack(row_idx + 1, current_rows, new_used, max_span_limit):
                    return True
                
                current_rows.pop()
                return False
            
            # Determine the range of possible next values
            # The next value must be > current_row[-1]
            # And current_row[-1] + (k+1 - pos) <= max_span_limit (rough bound)
            # Actually, the minimum span for remaining elements is not trivial,
            # but we can just iterate x from current_row[-1] + 1 to max_span_limit
            
            start_val = current_row[-1] + 1
            # Optimization: if we are at pos, we need k+1-pos more elements.
            # The smallest possible span for the rest is if we pick consecutive integers.
            # So current_row[-1] + (k+1-pos) must be <= max_span_limit.
            if current_row[-1] + (k + 1 - pos) > max_span_limit:
                return None
            
            for x in range(start_val, max_span_limit + 1):
                # Check differences with existing elements in current_row
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Recurse
                local_diffs_new = local_diffs | new_diffs
                if build_row(pos + 1, current_row + [x], local_diffs_new):
                    return True
                
            return None
        
        # Start building the row
        # To break symmetry, we can enforce that the new row's span is >= previous rows' spans?
        # Or just try to build it.
        
        # We can also prune if the number of available differences is insufficient.
        # Total differences needed for this row is k*(k+1)/2.
        # Total available differences up to max_span_limit is max_span_limit.
        # But they must be distinct from used_diffs.
        
        # Let's just call build_row
        build_row(1, [0], set())
        
        return False

    # We will try to improve the span
    # Start with the greedy solution's span
    current_limit = best_span - 1
    # We can try to find a solution with span <= current_limit
    # If found, update best_span and reduce current_limit
    
    # To make it faster, we can use a randomized restart or just a single deep search.
    # Given the small size, a single deep search with good pruning should work.
    
    # We'll run the backtracking search with the current best_span as the limit
    # If it finds a solution, it will update best_span.
    # We might need to run it multiple times with tighter limits.
    
    # Let's run it once with the greedy span as the limit.
    # If it finds a better one, it updates best_span.
    # Then we can try again with the new best_span.
    
    # To avoid infinite loops or too long runs, we'll just run it once with a tight limit
    # or use a heuristic.
    
    # Actually, the backtracking above is a bit complex with nonlocal state changes.
    # Let's simplify: just try to find ANY valid set of rows with span < best_span.
    
    # We'll use a simpler recursive function that returns a list of rows or None.
    
    def find_solution(row_idx, current_rows, used_diffs, max_span):
        if time.time() - start_time > time_limit:
            return None
        
        if row_idx == n_rows:
            return current_rows
        
        # Build next row
        row = [0]
        
        def build_row(pos, current_row, local_diffs):
            if time.time() - start_time > time_limit:
                return None
            
            if pos == k + 1:
                if current_row[-1] > max_span:
                    return None
                return current_row
            
            start_val = current_row[-1] + 1
            if current_row[-1] + (k + 1 - pos) > max_span:
                return None
            
            for x in range(start_val, max_span + 1):
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                res = build_row(pos + 1, current_row + [x], local_diffs | new_diffs)
                if res:
                    return res
            
            return None
        
        new_row = build_row(1, [0], set())
        if new_row:
            # Update used_diffs
            new_used = used_diffs.copy()
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    new_used.add(new_row[j] - new_row[i])
            
            res = find_solution(row_idx + 1, current_rows + [new_row], new_used, max_span)
            if res:
                return res
        
        return None

    # Try to find a solution with span < best_span
    # We'll try a few iterations, decreasing the limit
    # Start from best_span - 1 down to some lower bound
    # But this can be slow. Let's just try with best_span - 1 first.
    # If found, update and try again.
    
    # To be safe and fast, we'll just run the search once with the current best_span.
    # If it finds a solution, it means there is a solution with span <= best_span.
    # But we want < best_span. So we set limit = best_span - 1.
    
    # Let's try a range of limits
    # But given the time limit, we should be careful.
    
    # Alternative: Use a heuristic to generate candidate rows and then check compatibility.
    # This is often faster.
    
    # Let's stick with the backtracking but add a timeout check in the main loop.
    
    # We'll try to find a solution with span <= best_span - 1
    # If successful, update best_span and repeat.
    
    # To prevent infinite loops, we'll limit the number of attempts.
    attempts = 0
    while attempts < 5 and time.time() - start_time < time_limit:
        limit = best_span - 1
        if limit < 0:
            break
        res = find_solution(0, [], set(), limit)
        if res:
            best_rows = res
            best_span = max(max(r) for r in best_rows)
            attempts += 1
        else:
            break
            
    return best_rows
