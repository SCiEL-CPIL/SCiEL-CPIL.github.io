import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Calculate lower bound
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    M_lower = total_diffs
    
    # Upper bound: Greedy solution
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # We need k more elements
            # Simple greedy: pick smallest available x
            # This might be slow if we scan from 1 every time.
            # But M is small (~100).
            for _ in range(k):
                # Find smallest x > row[-1] such that x - row[i] not in used
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_M = max(max(r) for r in best_rows)
    
    # If greedy is already near lower bound, maybe just return it?
    # But let's try to improve with DFS.
    
    def solve_for_M(M):
        used = set()
        rows = []
        
        # Pre-check: Is it possible to fit the diffs?
        # We have M - 1 possible positive diffs? No, 1..M.
        # If total_diffs > M, impossible. (Handled by M_lower)
        
        def backtrack(r_idx, j_idx, row, last_val):
            if time.time() > deadline:
                return False
            
            if r_idx == n_rows:
                return True
            
            if j_idx == k + 1:
                rows.append(row)
                return backtrack(r_idx + 1, 0, [], 0)
            
            start_val = 1 if j_idx == 0 else last_val + 1
            # We must fill j_idx-th position (0-indexed)
            # Note: row[0] is 0.
            # If j_idx == 0, val must be 0.
            # Actually, let's handle row start separately.
            
            # Iteration over possible values for row[j_idx]
            # We want to try smaller values first to keep M small?
            # Actually M is fixed in this solver.
            # Trying smaller values might lead to "tight" rows which might block others?
            # Or it might be good.
            
            # Heuristic: Try values that generate diffs not in 'used'
            # Also, we must ensure that the remaining positions can be filled.
            # Max possible value for last element is M.
            
            # Range of values:
            # min: start_val
            # max: M
            # Also, we need to leave room for future elements?
            # If j_idx is k (last), val <= M.
            # If j_idx < k, val must be such that we can place k - j_idx more elements.
            # The smallest increment is 1. So max val = M - (k - j_idx).
            
            max_val = M - (k - j_idx)
            
            for val in range(start_val, max_val + 1):
                if j_idx == 0:
                    if val != 0:
                        continue
                    new_row = [0]
                    new_last = 0
                else:
                    if val <= last_val:
                        continue
                    new_row = row + [val]
                    new_last = val
                
                # Check collisions
                # Calculate diffs with previous elements in new_row
                # If j_idx == 0, no diffs.
                collides = False
                new_diffs = []
                for i in range(len(new_row) - 1):
                    d = val - new_row[i]
                    if d in used:
                        collides = True
                        break
                    new_diffs.append(d)
                
                if collides:
                    continue
                
                # Tentative update
                used.update(new_diffs)
                
                if backtrack(r_idx, j_idx + 1, new_row, new_last):
                    return True
                
                used.difference_update(new_diffs)
            
            return False
        
        return backtrack(0, 0, [], 0)

    # Search M from lower bound upwards
    # Limit the search range to avoid timeout
    # Greedy M is a good upper bound.
    # We search [M_lower, best_M]
    
    found = False
    for M in range(M_lower, best_M + 1):
        if time.time() > deadline - 5: # Safety margin
            break
        
        # Clear state? The function solve_for_M is self-contained.
        # But we need to capture the solution.
        # Let's modify solve_for_M to return the rows.
        
        def solve_for_M_with_result(M):
            used = set()
            rows = []
            
            def backtrack(r_idx, j_idx, row, last_val):
                if time.time() > deadline:
                    return False, None
                
                if r_idx == n_rows:
                    return True, rows[:]
                
                if j_idx == k + 1:
                    rows.append(row)
                    res, sol = backtrack(r_idx + 1, 0, [], 0)
                    if not res:
                        rows.pop()
                    return res, sol
                
                start_val = 1 if j_idx == 0 else last_val + 1
                max_val = M - (k - j_idx)
                
                for val in range(start_val, max_val + 1):
                    if j_idx == 0:
                        if val != 0:
                            continue
                        new_row = [0]
                        new_last = 0
                    else:
                        if val <= last_val:
                            continue
                        new_row = row + [val]
                        new_last = val
                    
                    collides = False
                    new_diffs = []
                    for i in range(len(new_row) - 1):
                        d = val - new_row[i]
                        if d in used:
                            collides = True
                            break
                        new_diffs.append(d)
                    
                    if collides:
                        continue
                    
                    used.update(new_diffs)
                    
                    res, sol = backtrack(r_idx, j_idx + 1, new_row, new_last)
                    if res:
                        return True, sol
                    
                    used.difference_update(new_diffs)
                
                return False, None
            
            return backtrack(0, 0, [], 0)

        res, sol = solve_for_M_with_result(M)
        if res:
            # Verify solution
            # (Assuming correctness of backtrack)
            return sol

    # If no better solution found, return greedy
    return best_rows
