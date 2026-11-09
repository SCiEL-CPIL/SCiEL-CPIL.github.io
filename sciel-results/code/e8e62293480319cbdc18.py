import time
import random
import sys

def construct(n_rows, k):
    row_size = k + 1
    diffs_per_row = row_size * (row_size - 1) // 2
    total_diffs = n_rows * diffs_per_row
    lb = total_diffs
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_rows = None
    best_max = float('inf')
    
    def is_valid(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def search_solution(M, time_limit):
        rows = []
        used_diffs = set()
        
        def backtrack(row_idx):
            if time.time() > time_limit:
                return None
            
            if row_idx == n_rows:
                return [r[:] for r in rows]
            
            current_row = [0]
            
            def build_row(elem_idx, current_used):
                if time.time() > time_limit:
                    return None
                
                if elem_idx == row_size:
                    rows.append(current_row[:])
                    res = backtrack(row_idx + 1)
                    if res:
                        return res
                    rows.pop()
                    return None
                
                last = current_row[-1]
                max_possible_last = M - (row_size - elem_idx)
                if last > max_possible_last:
                    return None
                
                candidates = list(range(last + 1, M + 1))
                random.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & current_used:
                        continue
                    
                    current_row.append(x)
                    new_used = current_used | new_diffs
                    res = build_row(elem_idx + 1, new_used)
                    if res:
                        return res
                    current_row.pop()
                    
            return build_row(1, used_diffs)
        
        restarts = 0
        while time.time() < time_limit:
            res = backtrack(0)
            if res:
                return res
            restarts += 1
            if restarts > 5000:
                break
        return None

    try:
        # 1. Find initial solution
        initial_M = 134
        sol = search_solution(initial_M, deadline - 10.0)
        if sol:
            best_rows = sol
            best_max = max(max(r) for r in sol)
        else:
            for M in range(135, 150):
                if time.time() > deadline: break
                sol = search_solution(M, deadline - 10.0)
                if sol:
                    best_rows = sol
                    best_max = max(max(r) for r in sol)
                    break
        
        if not best_rows:
            # Fallback greedy
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < row_size:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                rows.append(row)
            best_rows = rows
            best_max = max(max(r) for r in rows)

        # 2. Improve
        current_target = best_max - 1
        while current_target >= lb and time.time() < deadline:
            remaining = deadline - time.time()
            if remaining < 5:
                break
            budget = min(10.0, remaining * 0.8)
            
            sol = search_solution(current_target, time.time() + budget)
            if sol:
                best_rows = sol
                best_max = max(max(r) for r in sol)
                current_target = best_max - 1
            else:
                current_target -= 1

    except Exception:
        # Fallback on error
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_size:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows

    return best_rows
