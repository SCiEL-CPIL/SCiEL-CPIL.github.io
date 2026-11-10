import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 140
    
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

    # Initial solution
    current_rows = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_max = max(max(r) for r in best_rows)
    
    # Precompute used diffs for current state
    current_used = set()
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                current_used.add(r[j] - r[i])

    def get_max_entry(rows):
        return max(max(r) for r in rows)

    def try_move(r_idx, j_idx, new_val):
        row = current_rows[r_idx]
        old_val = row[j_idx]
        if new_val == old_val:
            return False
        if j_idx > 0 and new_val <= row[j_idx - 1]:
            return False
        if j_idx < k and new_val >= row[j_idx + 1]:
            return False
            
        old_diffs = set()
        new_diffs = set()
        for i in range(len(row)):
            if i == j_idx:
                continue
            if i < j_idx:
                old_diffs.add(old_val - row[i])
                new_diffs.add(new_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
                new_diffs.add(row[i] - new_val)
        
        for d in new_diffs:
            if d in current_used and d not in old_diffs:
                return False
        return True

    def apply_move(r_idx, j_idx, new_val):
        old_val = current_rows[r_idx][j_idx]
        row = current_rows[r_idx]
        for i in range(len(row)):
            if i == j_idx: continue
            if i < j_idx:
                current_used.discard(old_val - row[i])
                current_used.add(new_val - row[i])
            else:
                current_used.discard(row[i] - old_val)
                current_used.add(row[i] - new_val)
        row[j_idx] = new_val

    # Local Search
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            # Restart with a new greedy solution or perturb current
            # Perturb: randomize one row slightly
            # Or just restart greedy
            current_rows = greedy_construct()
            current_used = set()
            for r in current_rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        current_used.add(r[j] - r[i])
            no_improve_count = 0
            continue

        # Select move
        # 70%: Target the row with max entry, try to reduce its max
        # 30%: Random
        if random.random() < 0.7:
            max_row_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
            r_idx = max_row_idx
            j_idx = k # last element
            current_val = current_rows[r_idx][j_idx]
            lower = current_rows[r_idx][j_idx-1] + 1
            if lower >= current_val:
                # Try next largest element in that row?
                # For now, just skip or try random
                if random.random() < 0.5:
                    j_idx = random.randint(1, k)
                    r_idx = random.randint(0, n_rows-1)
                    current_val = current_rows[r_idx][j_idx]
                    lower = current_rows[r_idx][j_idx-1] + 1
                    upper = current_rows[r_idx][j_idx+1] - 1 if j_idx < k else current_val - 1
                    if lower > upper:
                        no_improve_count += 1
                        continue
                    new_val = random.randint(lower, min(upper, current_val - 1)) if current_val > lower else random.randint(lower, upper)
                else:
                    no_improve_count += 1
                    continue
            else:
                new_val = random.randint(lower, current_val - 1)
        else:
            r_idx = random.randint(0, n_rows - 1)
            j_idx = random.randint(1, k)
            current_val = current_rows[r_idx][j_idx]
            lower = current_rows[r_idx][j_idx-1] + 1
            upper = current_rows[r_idx][j_idx+1] - 1 if j_idx < k else current_val - 1
            if lower > upper:
                no_improve_count += 1
                continue
            # Bias towards smaller
            if current_val > lower and random.random() < 0.5:
                new_val = random.randint(lower, current_val - 1)
            else:
                new_val = random.randint(lower, upper)

        if try_move(r_idx, j_idx, new_val):
            apply_move(r_idx, j_idx, new_val)
            curr_max = get_max_entry(current_rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

    return best_rows
