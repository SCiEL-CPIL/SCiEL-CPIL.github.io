import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 140.0
    rows_len = k + 1
    
    # 1. Get initial solution via Greedy
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < rows_len:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    initial_sol = greedy()
    best_scope = max(max(row) for row in initial_sol)
    best_solution = initial_sol
    
    # 2. Try to improve with DFS
    # Symmetry breaking: max(row_i) <= max(row_{i+1})
    
    class Solver:
        def __init__(self):
            self.rows = []
            self.used = set()
            self.current_row = []
            self.row_diffs = set()
            
        def dfs(self, r_idx, pos, min_max):
            nonlocal best_scope, best_solution
            
            if time.time() > deadline:
                return False
            
            if pos == 0:
                self.current_row = [0]
                self.row_diffs = set()
                pos = 1
            
            if pos == rows_len:
                max_val = self.current_row[-1]
                
                # Symmetry check
                if max_val < min_max:
                    return False
                
                if r_idx == n_rows - 1:
                    if max_val < best_scope:
                        best_scope = max_val
                        best_solution = [r[:] for r in self.rows] + [self.current_row[:]]
                    return True
                else:
                    # Check global conflict
                    if not self.row_diffs.isdisjoint(self.used):
                        return False
                    
                    self.rows.append(self.current_row[:])
                    self.used.update(self.row_diffs)
                    
                    if self.dfs(r_idx + 1, 0, max_val):
                        return True
                        
                    self.used.difference_update(self.row_diffs)
                    self.rows.pop()
                    return False
            
            # Try next value
            start = self.current_row[-1] + 1
            
            # Pruning: if we are at the last position, val must be >= min_max
            # Also, val must be < best_scope to improve (or <= if we allow ties, but we want min)
            # If best_scope is the current limit, we only care about < best_scope.
            
            limit = best_scope - 1
            
            for val in range(start, limit + 1):
                if pos == rows_len - 1 and val < min_max:
                    continue
                
                valid = True
                new_diffs = set()
                for prev in self.current_row:
                    d = val - prev
                    if d in self.used or d in self.row_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                self.current_row.append(val)
                self.row_diffs.update(new_diffs)
                
                if self.dfs(r_idx, pos + 1, min_max):
                    return True
                
                self.current_row.pop()
                self.row_diffs.difference_update(new_diffs)
            
            return False

    solver = Solver()
    solver.dfs(0, 0, 0)
    
    return best_solution
