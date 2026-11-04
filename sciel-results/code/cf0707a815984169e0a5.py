import time
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    # Lower bound on max entry: need total_diffs_needed distinct differences in [1, M]
    # so M >= total_diffs_needed
    M_lower = total_diffs_needed
    
    def find_row(used_diffs, M, k, deadline):
        """Find a row of k+1 elements starting at 0 with all diffs unused, minimizing max element."""
        # Backtracking: build row element by element
        # row[0] = 0, then choose row[1], ..., row[k]
        best_row = None
        best_max = M + 1
        
        def backtrack(pos, current_row, max_elem):
            nonlocal best_row, best_max
            if time.time() > deadline:
                return
            if pos == k + 1:
                # Check all differences
                diffs = set()
                for i in range(k + 1):
                    for j in range(i + 1, k + 1):
                        d = current_row[j] - current_row[i]
                        if used_diffs[d]:
                            return
                        diffs.add(d)
                # All diffs are unused
                if max_elem < best_max:
                    best_max = max_elem
                    best_row = list(current_row)
                return
            
            # Pruning: if we've already exceeded best_max, skip
            if max_elem >= best_max:
                return
            
            # Try next element
            # Minimum possible next element is current_row[-1] + 1
            # Maximum we'd consider is best_max - 1 (to improve)
            if pos == 0:
                start_val = 0
            else:
                start_val = current_row[-1] + 1
            
            # Upper bound: we need room for remaining elements
            remaining = k - pos
            # The max element we can try: at most best_max - 1
            end_val = min(best_max, M + 1)  # exclusive upper bound
            
            for val in range(start_val, end_val):
                # Check differences with existing elements in current_row
                valid = True
                for existing in current_row:
                    d = val - existing
                    if d > M or used_diffs[d]:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_max = max(max_elem, val)
                if new_max >= best_max:
                    # If val >= best_max, no improvement possible
                    if val >= best_max:
                        break
                    continue
                
                current_row.append(val)
                backtrack(pos + 1, current_row, new_max)
                current_row.pop()
        
        backtrack(0, [0], 0)
        return best_row
    
    # Try increasing M values
    for M in range(M_lower, M_lower + 20):  # Try a reasonable range
        if time.time() > deadline:
            break
        
        # used_diffs[i] = True if difference i is already used
        used_diffs = [False] * (M + 1)
        rows = []
        
        success = True
        for r in range(n_rows):
            if time.time() > deadline:
                success = False
                break
            
            row = find_row(used_diffs, M, k, deadline)
            if row is None:
                success = False
                break
            
            # Mark all differences in this row as used
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used_diffs[d] = True
            
            rows.append(row)
        
        if success:
            return rows
    
    # Fallback: greedy approach (same as seed)
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
    return rows
