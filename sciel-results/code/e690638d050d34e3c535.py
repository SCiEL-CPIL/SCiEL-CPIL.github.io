import time
import random
import math

def construct(n_rows, k):
    deadline = time.time() + 115
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def greedy_build():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            max_val = 500 # Limit for greedy search
            for pos in range(1, k + 1):
                found = False
                start_val = row[-1] + 1
                for v in range(start_val, max_val):
                    valid_local = True
                    for prev in row:
                        d = v - prev
                        if d in used_diffs:
                            valid_local = False
                            break
                    if valid_local:
                        row.append(v)
                        for prev in row[:-1]:
                            used_diffs.add(v - prev)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    def optimize(rows):
        current_rows = [list(r) for r in rows]
        current_scope = get_scope(current_rows)
        best_rows_local = [list(r) for r in current_rows]
        best_scope_local = current_scope
        
        temp = 50.0
        cool = 0.95
        
        while temp > 0.1 and time.time() < deadline - 5:
            r_idx = random.randint(0, n_rows - 1)
            e_idx = random.randint(1, k)
            old_val = current_rows[r_idx][e_idx]
            
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            
            if new_val <= current_rows[r_idx][e_idx - 1]:
                continue
            if e_idx < k and new_val >= current_rows[r_idx][e_idx + 1]:
                continue
            
            current_rows[r_idx][e_idx] = new_val
            
            if is_valid(current_rows):
                new_scope = get_scope(current_rows)
                delta_scope = new_scope - current_scope
                if delta_scope < 0 or random.random() < math.exp(-delta_scope / temp):
                    current_scope = new_scope
                    if new_scope < best_scope_local:
                        best_scope_local = new_scope
                        best_rows_local = [list(r) for r in current_rows]
                else:
                    current_rows[r_idx][e_idx] = old_val
            else:
                current_rows[r_idx][e_idx] = old_val
                
            temp *= cool
            
        return best_rows_local, best_scope_local

    attempts = 0
    while time.time() < deadline - 5:
        rows = greedy_build()
        if rows is None:
            continue
            
        opt_rows, opt_scope = optimize(rows)
        
        if opt_scope < best_scope:
            best_scope = opt_scope
            best_rows = opt_rows
            
        attempts += 1
        if attempts > 100:
            break

    if best_rows is None:
        # Fallback: Construct a valid solution with large scope
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                v = row[-1] + 1
                while True:
                    valid = True
                    for prev in row:
                        d = v - prev
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        for prev in row[:-1]:
                            used_diffs.add(v - prev)
                        break
                    v += 1
            rows.append(row)
        return rows

    return best_rows
