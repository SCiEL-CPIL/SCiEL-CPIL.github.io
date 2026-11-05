import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 140  # 10s margin from 150s limit
    
    # Lower bound calculation
    needed_diffs = n_rows * (k + 1) * k // 2
    lb = int((-1 + (1 + 8 * needed_diffs) ** 0.5) / 2)
    lb = max(lb, 10)
    
    best_solution = None
    best_scope = float('inf')
    
    def greedy_construct(seed=0):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                max_try = 20
                for _ in range(max_try):
                    step = rng.randint(1, max(1, k * 2))
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            break
                        x += 1
            rows.append(row)
        return rows

    ub_rows = greedy_construct(42)
    ub_scope = max(max(r) for r in ub_rows)
    best_solution = ub_rows
    best_scope = ub_scope
    
    for M in range(lb, ub_scope):
        if time.time() > deadline - 10:
            break
            
        valid_rows = []
        
        def gen_rows(current, diffs_bitset, max_val):
            if len(current) == k + 1:
                valid_rows.append((tuple(current), diffs_bitset))
                return
            
            last = current[-1]
            for x in range(last + 1, max_val + 1):
                new_diffs_bitset = 0
                conflict = False
                for a in current:
                    d = x - a
                    bit = 1 << d
                    if diffs_bitset & bit:
                        conflict = True
                        break
                    new_diffs_bitset |= bit
                
                if not conflict:
                    current.append(x)
                    gen_rows(current, diffs_bitset | new_diffs_bitset, max_val)
                    current.pop()
                    
        gen_rows([0], 0, M)
        
        if not valid_rows:
            continue
            
        valid_rows.sort(key=lambda x: x[0])
        
        solution = [None] * n_rows
        found = False
        search_deadline = time.time() + 5
        
        def dfs(idx, used_bitset, start_idx):
            nonlocal found
            if time.time() > search_deadline:
                return False
                
            if idx == n_rows:
                found = True
                return True
                
            for i in range(start_idx, len(valid_rows)):
                row, bitset = valid_rows[i]
                if used_bitset & bitset:
                    continue
                    
                solution[idx] = row
                if dfs(idx + 1, used_bitset | bitset, i + 1):
                    return True
            return False
        
        dfs(0, 0, 0)
        
        if found:
            sol_rows = [list(r) for r in solution]
            current_scope = max(max(r) for r in sol_rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_solution = sol_rows
            return sol_rows
            
    return best_solution
