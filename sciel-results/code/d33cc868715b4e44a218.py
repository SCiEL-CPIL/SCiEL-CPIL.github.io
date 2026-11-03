import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    def greedy_construct(seed_val, strategy='compact'):
        """Construct rows using greedy with various strategies."""
        random.seed(seed_val)
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                elements_placed = len(row)
                if strategy == 'compact':
                    window = 30 + elements_placed * 10
                elif strategy == 'spread':
                    window = 50 + elements_placed * 15
                else:
                    window = 40 + elements_placed * 12
                
                search_end = min(last + window, lb + 200)
                
                candidates = list(range(last + 1, search_end + 1))
                
                if strategy == 'compact':
                    random.shuffle(candidates)
                    candidates.sort(key=lambda x: x + random.random() * 3)
                elif strategy == 'spread':
                    random.shuffle(candidates)
                else:
                    if random.random() < 0.5:
                        candidates.sort(key=lambda x: x + random.random() * 10)
                    else:
                        random.shuffle(candidates)
                
                for x in candidates:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    for x in range(last + 1, lb + 500):
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)
                        
                        if valid:
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                    if not found:
                        return None, float('inf')
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope
    
    # 1. Get a valid starting point using multiple greedy seeds
    initial_rows = None
    best_scope = float('inf')
    best_rows = None
    
    for s in range(50):
        rows, scope = greedy_construct(s, 'compact')
        if rows:
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
            if initial_rows is None:
                initial_rows = rows
        if time.time() > deadline - 100:
            break
    
    if initial_rows is None:
        # Fallback if greedy fails (unlikely)
        used = set()
        initial_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            initial_rows.append(row)
        best_rows = initial_rows
        best_scope = max(max(r) for r in rows)
    
    # 2. Simulated Annealing with diff_counts tracking
    current_rows = [list(r) for r in initial_rows]
    
    # Precompute diff counts
    diff_counts = {}
    current_conflict_count = 0
    
    def add_diff(d):
        nonlocal current_conflict_count
        if d in diff_counts:
            if diff_counts[d] == 1:
                current_conflict_count += 1
            diff_counts[d] += 1
        else:
            diff_counts[d] = 1
    
    def remove_diff(d):
        nonlocal current_conflict_count
        if d in diff_counts:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
            elif diff_counts[d] == 1:
                current_conflict_count -= 1
    
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                add_diff(d)
                
    current_scope = max(max(r) for r in current_rows)
    current_cost = current_scope + 1000 * current_conflict_count
    best_cost = current_cost
    best_rows_sa = [list(r) for r in current_rows]
    best_scope_sa = current_scope

    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline - 5:
        if T < T_min:
            T = 1.0
            cooling = 0.9995
            if steps > 50000:
                break
        
        # Generate Move
        move_type = random.random()
        
        if move_type < 0.7:
            # Value change in same row
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            p_idx = random.randint(1, len(row) - 1)
            
            old_val = row[p_idx]
            
            # Remove old_val's contributions
            diffs_to_remove = []
            for i in range(len(row)):
                if i != p_idx:
                    diffs_to_remove.append(abs(row[i] - old_val))
            
            for d in diffs_to_remove:
                remove_diff(d)
            
            row.pop(p_idx)
            
            # Try to find a new valid value
            found = False
            max_search = min(best_scope_sa + 10, 250)
            
            for _ in range(10):
                cand = random.randint(1, max_search)
                if cand in row:
                    continue
                
                valid = True
                new_diffs = []
                for a in row:
                    d = abs(cand - a)
                    if d in diff_counts:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    pos = 0
                    while pos < len(row) and row[pos] < cand:
                        pos += 1
                    row.insert(pos, cand)
                    for d in new_diffs:
                        add_diff(d)
                    found = True
                    break
            
            if not found:
                # Revert
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
            
            new_scope = max(max(r) for r in current_rows)
            new_cost = new_scope + 1000 * current_conflict_count
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                current_scope = new_scope
                if is_valid(current_rows) and current_cost < best_cost:
                    best_cost = current_cost
                    best_scope_sa = current_scope
                    best_rows_sa = [list(r) for r in current_rows]
            else:
                # Revert
                idx_new = row.index(cand)
                row.pop(idx_new)
                for a in row:
                    remove_diff(abs(a - cand))
                
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                
        else:
            # Move to another row
            r_idx = random.randint(0, n_rows - 1)
            new_r_idx = random.randint(0, n_rows - 1)
            while new_r_idx == r_idx:
                new_r_idx = random.randint(0, n_rows - 1)
            
            row = current_rows[r_idx]
            target_row = current_rows[new_r_idx]
            
            p_idx = random.randint(1, len(row) - 1)
            old_val = row[p_idx]
            
            # Remove old_val's contributions
            diffs_to_remove = []
            for i in range(len(row)):
                if i != p_idx:
                    diffs_to_remove.append(abs(row[i] - old_val))
            
            for d in diffs_to_remove:
                remove_diff(d)
            
            row.pop(p_idx)
            
            # Try to insert into target_row
            found = False
            valid = True
            new_diffs = []
            for a in target_row:
                d = abs(old_val - a)
                if d in diff_counts:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                pos = 0
                while pos < len(target_row) and target_row[pos] < old_val:
                    pos += 1
                target_row.insert(pos, old_val)
                for d in new_diffs:
                    add_diff(d)
                found = True
            
            if not found:
                # Revert
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
            
            new_scope = max(max(r) for r in current_rows)
            new_cost = new_scope + 1000 * current_conflict_count
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                current_scope = new_scope
                if is_valid(current_rows) and current_cost < best_cost:
                    best_cost = current_cost
                    best_scope_sa = current_scope
                    best_rows_sa = [list(r) for r in current_rows]
            else:
                # Revert
                idx_new = target_row.index(old_val)
                target_row.pop(idx_new)
                for a in target_row:
                    remove_diff(abs(a - old_val))
                
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
        
        T *= cooling
        steps += 1

    # 3. Final Compaction
    final_rows = [list(r) for r in best_rows_sa]
    used_set = get_used_diffs(final_rows)
    final_scope = max(max(r) for r in final_rows)
    
    for _ in range(100):
        if time.time() > deadline - 1:
            break
        row_idx = random.randint(0, n_rows - 1)
        row = final_rows[row_idx]
        if len(row) < 2:
            continue
        last = row[-1]
        if last <= 1:
            continue
        
        removed = row.pop()
        for a in row:
            used_set.discard(removed - a)
        
        found = False
        for x in range(last - 1, 0, -1):
            if x in row:
                continue
            valid = True
            for a in row:
                d = abs(x - a)
                if d == 0 or d in used_set:
                    valid = False
                    break
            if valid:
                pos = 0
                while pos < len(row) and row[pos] < x:
                    pos += 1
                row.insert(pos, x)
                for a in row:
                    if a != x:
                        used_set.add(abs(x - a))
                found = True
                break
        
        if not found:
            row.append(removed)
            row.sort()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_set.add(row[j] - row[i])
    
    final_scope = max(max(r) for r in final_rows)
    
    if is_valid(final_rows) and final_scope < best_scope_sa:
        best_rows_sa = final_rows
        best_scope_sa = final_scope
        
    # Final safety check
    if not is_valid(best_rows_sa):
        if is_valid(initial_rows):
            best_rows_sa = initial_rows
        else:
            used = set()
            best_rows_sa = []
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used:
                                valid = False
                                break
                        if valid:
                            break
                    row.append(x)
                    used.update({x - a for a in row if a != x})
                best_rows_sa.append(row)

    return best_rows_sa
