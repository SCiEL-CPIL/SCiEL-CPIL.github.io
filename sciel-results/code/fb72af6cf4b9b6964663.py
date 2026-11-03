import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def compute_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_init(n_rows, k):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_diffs = set()
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs) and not (new_diffs & local_diffs):
                    row.append(x)
                    local_diffs.update(new_diffs)
                x += 1
            rows.append(row)
            used_diffs.update(local_diffs)
        return rows, used_diffs

    def local_search(rows, used_diffs, n_rows, k, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            row_indices = list(range(n_rows))
            random.shuffle(row_indices)
            for i in row_indices:
                row = rows[i]
                for j in range(k, 0, -1):
                    old_val = row[j]
                    old_diffs = {old_val - a for a in row[:j]}
                    for d in old_diffs:
                        used_diffs.discard(d)
                    
                    start_x = row[j-1] + 1
                    found = False
                    for x in range(start_x, old_val):
                        new_diffs = {x - a for a in row[:j]}
                        if new_diffs & used_diffs:
                            continue
                        if not (new_diffs & used_diffs):
                            row[j] = x
                            used_diffs.update(new_diffs)
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        for d in old_diffs:
                            used_diffs.add(d)

        return rows, used_diffs

    def perturb(rows, used_diffs, n_rows, k, best_scope):
        i = random.randint(0, n_rows - 1)
        old_row = rows[i]
        old_diffs = compute_diffs(old_row)
        for d in old_diffs:
            used_diffs.discard(d)
        
        new_row = [0]
        local_diffs = set()
        max_val_hint = best_scope + 20
        for _ in range(k):
            candidates = []
            start = new_row[-1] + 1
            for x in range(start, max_val_hint + 1):
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & local_diffs:
                    continue
                candidates.append(x)
                if len(candidates) >= 50:
                    break
            
            if not candidates:
                rows[i] = old_row
                used_diffs.update(old_diffs)
                return rows, used_diffs
                
            if random.random() < 0.8:
                x = candidates[0]
            else:
                x = random.choice(candidates)
                
            new_row.append(x)
            for a in new_row[:-1]:
                local_diffs.add(x - a)
                
        rows[i] = new_row
        used_diffs.update(local_diffs)
        return rows, used_diffs

    best_rows = None
    best_scope = float('inf')
    
    rows, used_diffs = greedy_init(n_rows, k)
    rows, used_diffs = local_search(rows, used_diffs, n_rows, k, deadline)
    current_scope = max(max(r) for r in rows)
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [r[:] for r in rows]
        
    while time.time() < deadline:
        rows, used_diffs = perturb(rows, used_diffs, n_rows, k, best_scope)
        rows, used_diffs = local_search(rows, used_diffs, n_rows, k, deadline)
        current_scope = max(max(r) for r in rows)
        
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 105:
                break

    return best_rows
