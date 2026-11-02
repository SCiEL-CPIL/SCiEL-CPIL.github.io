import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def greedy_init():
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

    rows = greedy_init()
    
    row_diffs = []
    used_diffs = set()
    for r in rows:
        diffs = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                diffs.add(d)
        row_diffs.append(diffs)
        used_diffs.update(diffs)
        
    current_max = max(max(r) for r in rows)
    current_conflicts = 0
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    T = 10.0
    cooling = 0.9995
    
    def get_diffs_involving(row, j):
        diffs = []
        for i in range(j):
            diffs.append(row[j] - row[i])
        for i in range(j+1, len(row)):
            diffs.append(row[i] - row[j])
        return diffs

    while time.time() < deadline:
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k)
        
        row = rows[r_idx]
        old_val = row[j_idx]
        
        # Generate new value
        step = random.randint(-3, 3)
        new_val = old_val + step
        
        if j_idx > 0 and new_val <= row[j_idx-1]:
            continue
        if j_idx < k and new_val >= row[j_idx+1]:
            continue
        if new_val == old_val:
            continue

        old_diff_list = get_diffs_involving(row, j_idx)
        old_diffs = set(old_diff_list)
        
        # Tentative update
        used_diffs.difference_update(old_diffs)
        row_diffs[r_idx].difference_update(old_diffs)
        
        row[j_idx] = new_val
        
        new_diff_list = get_diffs_involving(row, j_idx)
        new_diffs = set(new_diff_list)
        
        # Check conflicts
        internal_conflicts = len(new_diff_list) - len(new_diffs)
        external_conflicts = len(new_diffs & used_diffs)
        conflicts = internal_conflicts + external_conflicts
        
        # Add new diffs to used_diffs tentatively
        used_diffs.update(new_diffs)
        row_diffs[r_idx].update(new_diffs)
        
        new_row_max = max(row)
        new_global_max = max(max(r) for r in rows)
        
        new_cost = new_global_max + 100 * conflicts
        old_cost = current_max + 100 * current_conflicts
        
        delta = new_cost - old_cost
        
        accept = False
        if delta <= 0:
            accept = True
        else:
            if random.random() < pow(2.718281828, -delta / T):
                accept = True
        
        if accept:
            current_max = new_global_max
            current_conflicts = conflicts
            if current_conflicts == 0 and current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Revert
            used_diffs.difference_update(new_diffs)
            row_diffs[r_idx].difference_update(new_diffs)
            
            row[j_idx] = old_val
            
            used_diffs.update(old_diffs)
            row_diffs[r_idx].update(old_diffs)
            
        T *= cooling
        if T < 0.01:
            T = 10.0 # Restart temperature to explore again

    return best_rows
