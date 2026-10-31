import time
import sys

def construct(n_rows, k):
    # n_rows=4, k=7 (8 numbers)
    # We want to minimize M (max entry)
    
    # Lower bound on M
    # Total diffs = n_rows * k * (k+1) / 2
    total_diffs = n_rows * k * (k + 1) // 2
    min_M = total_diffs # Since diffs are 1..M, M >= total_diffs
    
    # We will search M from min_M upwards
    # Time limit: 120s. 
    # We need to be careful not to spend too long on infeasible M.
    
    # Helper: Check if a row is valid given used diffs
    def check_row_validity(row, used_diffs):
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used_diffs or d in row_diffs:
                    return False
                row_diffs.add(d)
        return True, row_diffs

    # Backtracking to build a single row
    # row: list of numbers, currently [0]
    # used_diffs: set of diffs used by previous rows
    # target_len: k+1
    # max_val: M
    def build_row(row, used_diffs, target_len, max_val, time_limit):
        if len(row) == target_len:
            return [row] # Return list of solutions (just one for now)
        
        last_val = row[-1]
        # Try next value x > last_val
        # Heuristic: Try values that create "rare" diffs? 
        # Or just iterate x from last_val+1 to max_val
        
        # Optimization: 
        # The remaining slots are target_len - len(row) - 1
        # The max possible value is max_val.
        # We need to fit the remaining numbers.
        
        solutions = []
        for x in range(last_val + 1, max_val + 1):
            # Check if x can be added
            # New diffs: x - a for a in row
            new_diffs = {x - a for a in row}
            
            # Check against used_diffs and internal diffs
            # We need to track internal diffs of the partial row
            # This is inefficient if we recompute. 
            # Better to pass partial_diffs set.
            
            # Let's refine the function signature
            # build_row(row, partial_diffs, used_diffs, target_len, max_val)
            
            # For now, simple check:
            # If any d in new_diffs is in used_diffs, skip
            # If any d in new_diffs is in partial_diffs, skip
            
            # We need to maintain partial_diffs.
            
            pass

    # Revised Backtracking
    def build_row_backtrack(row, partial_diffs, used_diffs, target_len, max_val, start_time, time_limit):
        if time.time() - start_time > time_limit:
            return None # Timeout
        
        if len(row) == target_len:
            return list(row)
        
        last_val = row[-1]
        
        # Determine lower bound for next x
        # We need to place (target_len - len(row) - 1) more numbers after x.
        # The smallest they can be is x+1, x+2, ...
        # So x + (target_len - len(row)) <= max_val
        # x <= max_val - (target_len - len(row))
        
        max_next = max_val - (target_len - len(row))
        
        for x in range(last_val + 1, max_next + 1):
            new_diffs = {x - a for a in row}
            
            # Check conflicts
            conflict = False
            for d in new_diffs:
                if d in used_diffs or d in partial_diffs:
                    conflict = True
                    break
            
            if not conflict:
                # Add x
                row.append(x)
                partial_diffs.update(new_diffs)
                
                res = build_row_backtrack(row, partial_diffs, used_diffs, target_len, max_val, start_time, time_limit)
                if res is not None:
                    return res
                
                # Backtrack
                row.pop()
                partial_diffs.difference_update(new_diffs)
                
        return None

    def solve_for_M(M, start_time, time_limit):
        used_diffs = set()
        rows = []
        
        # Symmetry breaking: 
        # We can try to fix the first row to be the "greedy" one?
        # Or just build rows sequentially.
        # To reduce symmetry, we can require rows to be in lex order.
        # R_0 <= R_1 <= R_2 <= R_3
        
        def build_all_rows(idx, rows, used_diffs, prev_row, start_time, time_limit):
            if idx == n_rows:
                return list(rows)
            
            # Determine lower bound for first element of new row?
            # It must be >= prev_row (lex order)
            # But first element is always 0.
            # So we just need the whole row to be >= prev_row lex.
            
            # Build row
            row = [0]
            partial_diffs = set()
            
            # We need to ensure the resulting row is >= prev_row if idx > 0
            # This constraint is hard to enforce during backtracking of the row.
            # We can just generate the row and check at the end.
            # But that might generate many invalid rows.
            
            # Alternative: Don't use lex order symmetry breaking, just accept 24x search space.
            # Or, fix the first row to be the lexicographically smallest valid row?
            
            # Let's try without lex order first, but with a heuristic:
            # Always try to find a row.
            
            # We need to pass a constraint that this row must be distinct from previous ones?
            # The diff constraint handles that.
            
            # However, if we don't break symmetry, we might find the same set of rows in different orders.
            # That's fine, we just need one solution.
            
            res = build_row_backtrack(row, partial_diffs, used_diffs, k+1, M, start_time, time_limit)
            
            if res is None:
                return None
            
            # Check if this row is valid with used_diffs (already done in backtrack)
            # Update used_diffs
            for i in range(len(res)):
                for j in range(i+1, len(res)):
                    used_diffs.add(res[j] - res[i])
            
            rows.append(res)
            
            next_rows = build_all_rows(idx + 1, rows, used_diffs, res, start_time, time_limit)
            
            if next_rows is not None:
                return next_rows
            
            # Backtrack
            rows.pop()
            for i in range(len(res)):
                for j in range(i+1, len(res)):
                    used_diffs.remove(res[j] - res[i])
            
            return None

        return build_all_rows(0, [], used_diffs, None, start_time, time_limit)

    # Main loop
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    # Try M from min_M upwards
    # But we need to be careful. If M=112 is infeasible, it might take a long time to prove.
    # We should limit the time per M.
    
    # Heuristic: Start with a greedy solution to get an upper bound on M?
    # The current program does this.
    # Let's do a quick greedy to find a valid M_upper.
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_M = max(max(r) for r in best_rows)
    
    # Now search for better M
    # We know M is in [min_M, best_M]
    
    # We will try M values in increasing order.
    # To speed up, we can use the fact that if M is infeasible, M+1 might be too.
    
    # However, backtracking can be slow.
    # We should use a time slice for each M.
    
    # If we have time, we try M = min_M, min_M+1, ...
    
    # Optimization: 
    # The number of rows is small (4).
    # The branching factor in build_row is limited by M.
    
    # Let's try a few M values.
    
    # If time is running out, return best_rows found so far.
    
    # We can also try to improve the greedy solution with local search if CSP fails or takes too long.
    # But the directive says to use CSP.
    
    # Let's implement the CSP solver carefully.
    
    # One issue: build_row_backtrack might be slow if M is large.
    # But M is around 112.
    
    # We will iterate M from min_M to best_M - 1.
    # If we find a solution, update best_M and best_rows.
    # If we exhaust the search for an M (return None), move to next M.
    
    # To avoid hanging on infeasible M, we set a time limit per M.
    # e.g., 5 seconds per M.
    
    # But if min_M is feasible, we want to find it quickly.
    
    # Let's go.
    
    current_best_rows = best_rows
    current_best_M = best_M
    
    # We'll try M in range(min_M, current_best_M)
    # But if current_best_M is close to min_M, there are few to try.
    
    # If greedy found 134, and min_M is 112, we have 22 values to try.
    # 5s each = 110s. Fits in budget.
    
    # However, if 112 is infeasible, it might take the full 5s to prove.
    
    # Let's start.
    
    for M in range(min_M, current_best_M):
        if time.time() > deadline - 10:
            break
            
        # Try to solve for this M
        # We need to reset the state
        # The solve_for_M function uses global-ish state (used_diffs) but we pass it in.
        # But build_row_backtrack uses recursion.
        
        # We need to be careful with the time limit inside the recursion.
        
        res = solve_for_M(M, time.time(), 5.0) # 5s limit per M
        
        if res is not None:
            current_best_rows = res
            current_best_M = M
            # Since we are searching upwards, if we find a solution for M, 
            # it's the best so far. 
            # But we should check if we can find an even smaller one? 
            # No, we started from min_M. So the first one found is the minimum.
            # So we can break?
            # Yes, if we search M in increasing order, the first valid M is the minimum.
            break
            
    return current_best_rows
