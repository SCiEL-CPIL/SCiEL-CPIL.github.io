import random
import time

def construct(n_rows, k):
    num_elements = k + 1  # 6
    
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Greedy initial
    def greedy_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    rows = greedy_initial()
    best_rows = [row[:] for row in rows]
    best_scope = max(max(row) for row in rows)
    
    start_time = time.time()
    time_limit = 110
    
    # Precompute diffs for each row
    row_diffs = [get_diffs(row) for row in rows]
    
    # Local search
    while time.time() - start_time < time_limit:
        r = random.randint(0, n_rows - 1)
        
        other_diffs = set()
        for idx in range(n_rows):
            if idx != r:
                other_diffs |= row_diffs[idx]
        
        current_row = rows[r]
        current_max = max(current_row)
        
        found = None
        for target_max in range(current_max - 1, max(num_elements - 1, 0), -1):
            row = [0]
            
            def backtrack(pos):
                if pos == num_elements:
                    return row[:]
                start = row[-1] + 1
                end = min(target_max, start + 30)
                candidates = list(range(start, end + 1))
                random.shuffle(candidates)
                for v in candidates:
                    new_diffs = {v - a for a in row}
                    if new_diffs & other_diffs:
                        continue
                    row.append(v)
                    result = backtrack(pos + 1)
                    if result is not None:
                        return result
                    row.pop()
                return None
            
            found = backtrack(1)
            if found is not None:
                break
        
        if found is not None:
            rows[r] = found
            row_diffs[r] = get_diffs(found)
            current_scope = max(max(row) for row in rows)
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [row[:] for row in rows]
    
    return best_rows
