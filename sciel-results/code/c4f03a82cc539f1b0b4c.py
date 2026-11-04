import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 140.0
    MAX_DIFF = 5000  # Max possible difference we track in bitmask
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def diff_bitmask(row):
        """Compute bitmask of all differences in a row."""
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= MAX_DIFF:
                    mask |= (1 << d)
        return mask
    
    def row_diffs_list(row):
        """Get list of all differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    def greedy_init():
        """Simple deterministic greedy: pick smallest valid next element."""
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_mask = 0
                    valid = True
                    for a in row:
                        d = x - a
                        if d > MAX_DIFF:
                            valid = False
                            break
                        new_mask |= (1 << d)
                    if not valid:
                        x += 1
                        continue
                    if new_mask & used_mask:
                        x += 1
                        continue
                    # Check internal consistency (should be fine since x > all in row)
                    row.append(x)
                    used_mask |= new_mask
                    break
            rows.append(row)
        return rows
    
    def randomized_greedy(seed=None):
        """Randomized greedy with diverse candidate selection."""
        if seed is not None:
            random.seed(seed)
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                candidates = []
                x = row[-1] + 1
                # Search up to a reasonable bound
                bound = max(row[-1] * 2 + 100, 300)
                while x <= bound:
                    new_mask = 0
                    valid = True
                    for a in row:
                        d = x - a
                        if d > MAX_DIFF:
                            valid = False
                            break
                        new_mask |= (1 << d)
                    if not valid:
                        x += 1
                        continue
                    if not (new_mask & used_mask):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Expand search
                    bound *= 2
                    x = row[-1] + 1
                    while x <= bound:
                        new_mask = 0
                        valid = True
                        for a in row:
                            d = x - a
                            if d > MAX_DIFF:
                                valid = False
                                break
                            new_mask |= (1 << d)
                        if not valid:
                            x += 1
                            continue
                        if not (new_mask & used_mask):
                            candidates.append(x)
                        x += 1
                
                if not candidates:
                    # Fallback: find first valid
                    x = row[-1] + 1
                    while True:
                        new_mask = 0
                        valid = True
                        for a in row:
                            d = x - a
                            if d > MAX_DIFF:
                                valid = False
                                break
                            new_mask |= (1 << d)
                        if not valid:
                            x += 1
                            continue
                        if not (new_mask & used_mask):
                            candidates = [x]
                            break
                        x += 1
                
                # Randomized selection with bias toward smaller values
                if len(candidates) > 1:
                    if random.random() < 0.6:
                        top_n = min(5, len(candidates))
                        val = candidates[random.randint(0, top_n - 1)]
                    elif random.random() < 0.5:
                        val = candidates[random.randint(0, len(candidates) // 2)]
                    else:
                        val = random.choice(candidates)
                else:
                    val = candidates[0]
                
                row.append(val)
                new_mask = 0
                for a in row[:-1]:
                    d = val - a
                    new_mask |= (1 << d)
                used_mask |= new_mask
            rows.append(row)
        return rows
    
    def local_search(rows, time_limit=30.0):
        """Local search: try to reduce max entry by moving individual elements."""
        current = [r[:] for r in rows]
        current_max = max_entry(current)
        
        # Build global diff mask
        global_mask = 0
        for row in current:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= MAX_DIFF:
                        global_mask |= (1 << d)
        
        positions = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            improved = False
            random.shuffle(positions)
            
            for r, i in positions:
                if time.time() - start_time >= time_limit:
                    break
                
                row = current[r]
                old_val = row[i]
                lower = row[i - 1] + 1
                upper = row[i + 1] - 1 if i < k else old_val
                
                # Compute diffs involving position i
                old_diffs = []
                for j in range(len(row)):
                    if j == i:
                        continue
                    if j < i:
                        old_diffs.append(old_val - row[j])
                    else:
                        old_diffs.append(row[j] - old_val)
                
                # Remove old diffs from global mask
                temp_mask = global_mask
                for d in old_diffs:
                    if d <= MAX_DIFF:
                        temp_mask &= ~(1 << d)
                
                # Try to move to a smaller value (to reduce max if it's the max)
                # Or try values that keep max the same but might enable future improvements
                best_new_val = None
                best_new_mask = None
                
                # If this is the max element, try to reduce it
                if old_val == current_max:
                    for new_val in range(lower, old_val):
                        new_diffs = []
                        conflict = False
                        for j in range(len(row)):
                            if j == i:
                                continue
                            if j < i:
                                nd = new_val - row[j]
                            else:
                                nd = row[j] - new_val
                            if nd > MAX_DIFF:
                                conflict = True
                                break
                            if temp_mask & (1 << nd):
                                conflict = True
                                break
                            new_diffs.append(nd)
                        
                        if conflict:
                            continue
                        
                        # Check internal consistency
                        new_mask = temp_mask
                        for nd in new_diffs:
                            new_mask |= (1 << nd)
                        
                        # Check if this reduces max
                        new_row_max = max(row)
                        if new_val < old_val:
                            # Find new max of this row
                            new_row_max = max(new_val, max(row[j] for j in range(len(row)) if j != i))
                        total_max = max(max(row2) if r2 != r else new_row_max 
                                       for r2, row2 in enumerate(current))
                        
                        if total_max < current_max:
                            best_new_val = new_val
                            best_new_mask = new_mask
                            break
                else:
                    # For non-max elements, try a range of values
                    search_range = min(upper - lower + 1, 50)
                    for new_val in range(lower, lower + search_range):
                        if new_val == old_val:
                            continue
                        new_diffs = []
                        conflict = False
                        for j in range(len(row)):
                            if j == i:
                                continue
                            if j < i:
                                nd = new_val - row[j]
                            else:
                                nd = row[j] - new_val
                            if nd > MAX_DIFF:
                                conflict = True
                                break
                            if temp_mask & (1 << nd):
                                conflict = True
                                break
                            new_diffs.append(nd)
                        
                        if conflict:
                            continue
                        
                        new_mask = temp_mask
                        for nd in new_diffs:
                            new_mask |= (1 << nd)
                        
                        # Only accept if it doesn't increase max
                        new_row_max = max(row)
                        if new_val > old_val:
                            new_row_max = max(new_val, max(row[j] for j in range(len(row)) if j != i))
                        total_max = max(max(row2) if r2 != r else new_row_max
                                       for r2, row2 in enumerate(current))
                        
                        if total_max <= current_max:
                            best_new_val = new_val
                            best_new_mask = new_mask
                            break
                
                if best_new_val is not None:
                    current[r][i] = best_new_val
                    global_mask = best_new_mask
                    current_max = max_entry(current)
                    improved = True
            
            if not improved:
                # No improvement in this pass, try a few more random shuffles
                for _ in range(3):
                    if time.time() - start_time >= time_limit:
                        break
                    random.shuffle(positions)
                    found = False
                    for r, i in positions:
                        row = current[r]
                        old_val = row[i]
                        lower = row[i - 1] + 1
                        upper = row[i + 1] - 1 if i < k else old_val
                        
                        old_diffs = []
                        for j in range(len(row)):
                            if j == i:
                                continue
                            if j < i:
                                old_diffs.append(old_val - row[j])
                            else:
                                old_diffs.append(row[j] - old_val)
                        
                        temp_mask = global_mask
                        for d in old_diffs:
                            if d <= MAX_DIFF:
                                temp_mask &= ~(1 << d)
                        
                        if old_val == current_max:
                            for new_val in range(lower, old_val):
                                new_diffs = []
                                conflict = False
                                for j in range(len(row)):
                                    if j == i:
                                        continue
                                    if j < i:
                                        nd = new_val - row[j]
                                    else:
                                        nd = row[j] - new_val
                                    if nd > MAX_DIFF:
                                        conflict = True
                                        break
                                    if temp_mask & (1 << nd):
                                        conflict = True
                                        break
                                    new_diffs.append(nd)
                                
                                if conflict:
                                    continue
                                
                                new_mask = temp_mask
                                for nd in new_diffs:
                                    new_mask |= (1 << nd)
                                
                                new_row_max = max(new_val, max(row[j] for j in range(len(row)) if j != i))
                                total_max = max(max(row2) if r2 != r else new_row_max
                                               for r2, row2 in enumerate(current))
                                
                                if total_max < current_max:
                                    current[r][i] = new_val
                                    global_mask = new_mask
                                    current_max = total_max
                                    found = True
                                    break
                        else:
                            for new_val in range(lower, min(upper + 1, lower + 30)):
                                if new_val == old_val:
                                    continue
                                new_diffs = []
                                conflict = False
                                for j in range(len(row)):
                                    if j == i:
                                        continue
                                    if j < i:
                                        nd = new_val - row[j]
                                    else:
                                        nd = row[j] - new_val
                                    if nd > MAX_DIFF:
                                        conflict = True
                                        break
                                    if temp_mask & (1 << nd):
                                        conflict = True
                                        break
                                    new_diffs.append(nd)
                                
                                if conflict:
                                    continue
                                
                                new_mask = temp_mask
                                for nd in new_diffs:
                                    new_mask |= (1 << nd)
                                
                                new_row_max = max(row)
                                if new_val > old_val:
                                    new_row_max = max(new_val, max(row[j] for j in range(len(row)) if j != i))
                                total_max = max(max(row2) if r2 != r else new_row_max
                                               for r2, row2 in enumerate(current))
                                
                                if total_max <= current_max:
                                    current[r][i] = new_val
                                    global_mask = new_mask
                                    current_max = total_max
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        break
        
        return current, current_max
    
    def sa_polish(rows, time_limit=25.0):
        """Simulated annealing to polish the solution."""
        current = [r[:] for r in rows]
        current_max = max_entry(current)
        
        # Build diff count structure
        diff_counts = {}
        for row in current:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        def get_violations():
            return sum(c - 1 for c in diff_counts.values() if c > 1)
        
        current_violations = get_violations()
        current_cost = current_violations * 1000000 + current_max
        
        T = 50.0
        T_min = 0.1
        cooling = 0.99995
        
        indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
            
            r, i = random.choice(indices)
            row = current[r]
            old_val = row[i]
            lower = row[i - 1] + 1
            upper = row[i + 1] - 1 if i < k else row[i] + 50
            
            if random.random() < 0.5:
                span = max(1, upper - lower + 1)
                new_val = lower + int(random.random() * span * 0.7)
            else:
                new_val = random.randint(lower, upper)
            
            if new_val == old_val or new_val < lower or (i < k and new_val > upper):
                continue
            
            removed_diffs = []
            added_diffs = []
            for j in range(len(row)):
                if j == i:
                    continue
                if j < i:
                    removed_diffs.append(old_val - row[j])
                    added_diffs.append(new_val - row[j])
                else:
                    removed_diffs.append(row[j] - old_val)
                    added_diffs.append(row[j] - new_val)
            
            for d in removed_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in added_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_violations = get_violations()
            new_max = max(max(row) for row in current)
            new_cost = new_violations * 1000000 + new_max
            
            delta = new_cost - current_cost
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 0.01)):
                current[r][i] = new_val
                current_violations = new_violations
                current_cost = new_cost
                if new_max < current_max and new_violations == 0:
                    current_max = new_max
            else:
                for d in added_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in removed_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                current[r][i] = old_val
            
            T *= cooling
        
        return current, current_max
    
    # === Main strategy ===
    # Phase 1: Generate diverse starts
    best_rows = greedy_init()
    best_max = max_entry(best_rows)
    
    n_restarts = 10
    per_restart_time = 8.0
    start_phase = time.time()
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        seed = restart * 12345 + 67890
        rows = randomized_greedy(seed=seed)
        cur_max = max_entry(rows)
        
        if cur_max <= best_max + 30:
            ls_time = min(per_restart_time, deadline - time.time() - 15.0)
            if ls_time > 3.0:
                refined, refined_max = local_search(rows, time_limit=ls_time)
                if refined_max < best_max and is_valid(refined):
                    best_rows = refined
                    best_max = refined_max
    
    # Phase 2: Final local search on the best
    remaining = deadline - time.time() - 30.0
    if remaining > 5.0:
        refined, refined_max = local_search(best_rows, time_limit=min(remaining, 40.0))
        if refined_max < best_max and is_valid(refined):
            best_rows = refined
            best_max = refined_max
    
    # Phase 3: SA polish
    remaining = deadline - time.time() - 5.0
    if remaining > 3.0:
        polished, polished_max = sa_polish(best_rows, time_limit=min(remaining, 25.0))
        if polished_max < best_max and is_valid(polished):
            best_rows = polished
            best_max = polished_max
    
    # Final validation
    if not is_valid(best_rows):
        best_rows = greedy_init()
    
    return best_rows
