import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Greedy baseline
    def greedy_construct():
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
    
    best_solution = greedy_construct()
    best_scope = max(max(r) for r in best_solution)
    
    # Backtracking solver
    def solve(max_val, timeout):
        used = [0] * (max_val + 1)
        rows = []
        
        def backtrack(row_idx, row, col_idx):
            if time.time() > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            if col_idx == k + 1:
                rows.append(row[:])
                result = backtrack(row_idx + 1, [0], 0)
                if not result:
                    rows.pop()
                return result
            
            start_val = row[-1] + 1 if row else 1
            
            for val in range(start_val, max_val + 1):
                if time.time() > timeout:
                    return False
                
                new_diffs = []
                valid = True
                for a in row:
                    d = val - a
                    if d > max_val or used[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                for d in new_diffs:
                    used[d] = 1
                row.append(val)
                
                if backtrack(row_idx, row, col_idx + 1):
                    return True
                
                row.pop()
                for d in new_diffs:
                    used[d] = 0
            
            return False
        
        success = backtrack(0, [0], 0)
        if success:
            return rows
        return None
    
    # Try to find better solutions
    # Work downward from best_scope - 1
    for scope in range(best_scope - 1, 10, -1):
        if time.time() > deadline - 15.0:
            break
        
        # Theoretical lower bound check
        total_needed = n_rows * k * (k + 1) // 2
        if scope < total_needed:
            break
        if scope < k * (k + 1) // 2:
            break
        
        result = solve(scope, deadline - 10.0)
        if result is not None:
            new_scope = max(max(r) for r in result)
            if new_scope < best_scope:
                best_scope = new_scope
                best_solution = result
            break
    
    return best_solution
