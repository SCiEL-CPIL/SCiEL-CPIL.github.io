import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 110
    
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    
    greedy_rows = _greedy_construct(n_rows, k)
    if not greedy_rows:
        return []
    upper_bound = max(max(r) for r in greedy_rows)
    
    best_solution = greedy_rows
    
    for M in range(lower_bound, upper_bound + 1):
        if time.time() > deadline - 5:
            break
            
        solution = _backtrack_solve(n_rows, k, M, deadline)
        if solution:
            return solution
            
    return best_solution

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _backtrack_solve(n_rows, k, max_scope, deadline):
    
    def time_check():
        return time.time() < deadline

    def _solve(rows, current_row, used_diffs, prev_row):
        if not time_check():
            return None
            
        if len(rows) == n_rows:
            if len(current_row) == k + 1:
                # Final check for symmetry: current_row >= prev_row
                if prev_row is None or _lex_ge(current_row, prev_row):
                    return rows + [current_row]
            return None
            
        if len(current_row) == k + 1:
            # Check symmetry before committing
            if prev_row is not None and not _lex_ge(current_row, prev_row):
                return None
                
            new_rows = rows + [current_row]
            new_current = [0]
            return _solve(new_rows, new_current, used_diffs, current_row)
            
        # Extend current row
        last = current_row[-1]
        elements_needed = (k + 1) - len(current_row)
        
        if last + elements_needed > max_scope:
            return None
            
        # Symmetry pruning:
        # If we are in the middle of constructing a row, and prev_row exists,
        # we can prune if current_row is already strictly less than prev_row
        # at the first position where they differ.
        # Actually, if current_row[j] < prev_row[j], we can prune.
        # If current_row[j] > prev_row[j], we are "ahead", so no further symmetry constraint for this row.
        # If current_row[j] == prev_row[j], we must ensure the rest doesn't fall behind.
        
        # Let's determine the minimum allowed value for the next element based on symmetry.
        min_next = last + 1
        
        if prev_row is not None:
            # Find the first index where current_row and prev_row differ
            # Or if they are equal so far, we are tied.
            # If current_row is a prefix of prev_row (and shorter), we are tied so far.
            # We need current_row to be >= prev_row.
            
            # If len(current_row) > 0:
            # Check the last element we just added? No, we are about to add one.
            # Let's check the current state.
            
            # If current_row is already strictly less than prev_row at any position i < len(current_row), prune.
            # This check is done when we pick a value.
            
            pass # Logic handled in loop

        for x in range(min_next, max_scope + 1):
            if not time_check():
                return None
                
            # Symmetry check:
            # If prev_row exists:
            if prev_row is not None:
                # Check if adding x keeps the row >= prev_row
                # Compare current_row + [x] with prev_row
                # If at any index i, new_row[i] < prev_row[i], invalid.
                # If at any index i, new_row[i] > prev_row[i], valid (for symmetry).
                # If all equal so far, continue.
                
                # Optimization:
                # If we have already "won" the lex order (some previous element was larger), no constraint.
                # We can track a flag `is_greater` or just check.
                
                # Simple check:
                # Create new_row = current_row + [x]
                # If not _lex_ge(new_row, prev_row): continue
                
                # To avoid creating list, check manually:
                # Let new_row be current_row + [x]
                # For i in range(len(current_row)):
                #    if current_row[i] < prev_row[i]: invalid
                #    if current_row[i] > prev_row[i]: break (valid)
                # At index len(current_row):
                #    if x < prev_row[len(current_row)]: invalid
                #    if x > prev_row[len(current_row)]: valid
                #    if x == prev_row[len(current_row)]: tied, continue to next element (but this is the last check for this step)
                
                # Actually, if we are tied up to index len(current_row)-1, and x < prev_row[len(current_row)], we prune.
                
                # Let's implement a helper or inline logic.
                
                # Check prefix
                is_greater = False
                for i in range(len(current_row)):
                    if current_row[i] > prev_row[i]:
                        is_greater = True
                        break
                    elif current_row[i] < prev_row[i]:
                        # Already less, should have been pruned earlier?
                        # Yes, if we enforced it at each step.
                        # But if we didn't, we prune now.
                        break
                
                if not is_greater:
                    # We are tied or less up to here.
                    # If less, prune.
                    # We need to check if we are less.
                    # If any current_row[i] < prev_row[i], prune.
                    for i in range(len(current_row)):
                        if current_row[i] < prev_row[i]:
                            break # Prune this x
                        if current_row[i] > prev_row[i]:
                            break # Greater, so x can be anything? No, if greater, the row is already >=.
                    # If we are tied so far:
                    else:
                        # Check x against prev_row at this position
                        if x < prev_row[len(current_row)]:
                            break # Prune
                            
            # Validity check for differences
            new_diffs = []
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            new_current = current_row + [x]
            new_used = used_diffs.union(new_diffs)
            
            result = _solve(rows, new_current, new_used, prev_row)
            if result:
                return result
                
        return None

    return _solve([], [0], set(), None)

def _lex_ge(a, b):
    # Returns True if a >= b lexicographically
    # a and b have same length (k+1)
    for i in range(len(a)):
        if a[i] > b[i]:
            return True
        elif a[i] < b[i]:
            return False
    return True # Equal
