import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

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

    # Greedy construction
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_row_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                forbidden = used | current_row_diffs
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        row.append(x)
                        current_row_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
            used |= current_row_diffs
        return rows

    # Bitset-accelerated row generation
    def generate_row_bitset(used_diffs, k_val, max_scope, randomize=False, p_smallest=0.7, window=5):
        """Generate a single row using bitset for fast conflict checking."""
        max_d = max_scope + 1
        forbidden_bits = 0
        for d in used_diffs:
            if d < max_d:
                forbidden_bits |= (1 << d)
        
        row = [0]
        row_diffs_bits = 0
        
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            
            while x < max_scope:
                conflict = False
                for a in row:
                    d = x - a
                    if d < max_d and (forbidden_bits >> d) & 1:
                        conflict = True
                        break
                    if d < max_d and (row_diffs_bits >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            
            if not candidates:
                return None
            
            if randomize:
                if random.random() < p_smallest:
                    next_x = candidates[0]
                else:
                    next_x = random.choice(candidates)
            else:
                next_x = candidates[0]
            
            for a in row:
                d = next_x - a
                if d < max_d:
                    row_diffs_bits |= (1 << d)
            
            row.append(next_x)
        
        return row

    # Local compaction: try to move each element to a smaller position
    def local_compact(rows, used_diffs, max_iterations=1000):
        current_scope = max(max(r) for r in rows)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for p_idx in range(len(row) - 1, 0, -1):
                    old_val = row[p_idx]
                    if old_val <= 1:
                        continue
                    
                    row.pop(p_idx)
                    
                    diffs_to_remove = set()
                    for a in row:
                        d = abs(old_val - a)
                        diffs_to_remove.add(d)
                    used_diffs -= diffs_to_remove
                    
                    found = False
                    for x in range(old_val - 1, 0, -1):
                        if x in row:
                            continue
                        valid = True
                        for a in row:
                            d = abs(x - a)
                            if d == 0 or d in used_diffs:
                                valid = False
                                break
                        if valid:
                            pos = 0
                            while pos < len(row) and row[pos] < x:
                                pos += 1
                            row.insert(pos, x)
                            for a in row:
                                if a != x:
                                    used_diffs.add(abs(x - a))
                            found = True
                            improved = True
                            break
                    
                    if not found:
                        pos = 0
                        while pos < len(row) and row[pos] < old_val:
                            pos += 1
                        row.insert(pos, old_val)
                        for a in row:
                            if a != old_val:
                                used_diffs.add(abs(old_val - a))
        
        return rows, used_diffs

    # Bitset-accelerated backtracking scope search
    def backtracking_scope_search(n_rows, k, target_scope, deadline, seed_offset=0):
        """
        Try to find a solution with max entry <= target_scope using backtracking.
        Returns rows if found, None otherwise.
        """
        rng = random.Random(seed_offset * 7919 + 31)
        L = k + 1
        rows = []
        
        used_diffs = [False] * (target_scope + 1)
        used_count = 0

        def build_row(pos, current_mark, row_so_far):
            nonlocal used_count
            if time.time() > deadline:
                return None

            if pos == L:
                return list(row_so_far)

            # Prune: need at least (L - pos - 1) more values
            max_x = target_scope - (L - pos - 1)
            if max_x <= current_mark:
                return None

            # Find valid candidates with early termination
            valid_candidates = []
            for x in range(current_mark + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if d <= target_scope and used_diffs[d]:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
                if len(valid_candidates) >= 40:
                    break

            if not valid_candidates:
                return None

            # Bias towards smaller values but allow some randomness
            n_to_try = min(8, len(valid_candidates))
            candidates_to_try = valid_candidates[:n_to_try]
            if len(valid_candidates) > n_to_try and rng.random() < 0.3:
                candidates_to_try.append(rng.choice(valid_candidates[n_to_try:]))
            
            # Shuffle slightly for diversity
            rng.shuffle(candidates_to_try)

            for x in candidates_to_try:
                # Check all differences
                new_diffs = []
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if d <= target_scope and used_diffs[d]:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue

                # Apply
                for d in new_diffs:
                    used_diffs[d] = True
                    used_count += 1
                row_so_far.append(x)

                result = build_row(pos + 1, x, row_so_far)
                if result is not None:
                    return result

                # Undo
                row_so_far.pop()
                for d in new_diffs:
                    used_diffs[d] = False
                    used_count -= 1

            return None

        for _ in range(n_rows):
            if time.time() > deadline:
                return None
            row_result = build_row(0, -1, [0])
            if row_result is None:
                return None
            rows.append(row_result)
        
        return rows

    # 1. Initial Solution: Deterministic Greedy
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # 2. Local Compaction to tighten the initial solution
    used = get_used_diffs(rows)
    rows, used = local_compact(rows, used)
    compacted_max = max(max(r) for r in rows)
    if compacted_max < best_max:
        best_max = compacted_max
        best_rows = [r[:] for r in rows]
    current_max = best_max

    # 3. Try backtracking scope search for progressively smaller scopes
    for target in range(best_max - 1, max(best_max - 10, 100), -1):
        if time.time() > deadline - 40:
            break
        result = backtracking_scope_search(n_rows, k, target, deadline - 40, seed_offset=target)
        if result is not None and is_valid(result):
            best_rows = [r[:] for r in result]
            best_max = target
            current_max = target
            break

    # 4. Simulated Annealing with Row Regeneration
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    rows = [r[:] for r in best_rows]
    used = get_used_diffs(rows)
    
    while time.time() < deadline - 15:
        if T < T_min:
            T = 5.0
            cooling = 0.998
            
        r_idx = random.randint(0, n_rows - 1)
        
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        
        used -= old_diffs
        
        new_row = generate_row_bitset(used, k, best_max + 20, randomize=True, p_smallest=0.8, window=3)
        
        if new_row is None:
            used |= old_diffs
            continue
        
        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)
        
        delta = new_global_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            rows[r_idx] = new_row
            used |= set(new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i + 1, len(new_row)))
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            used |= old_diffs
            
        T *= cooling

    # 5. Final Local Search: Individual Element Moves
    current_rows = [list(r) for r in best_rows]
    
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
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                add_diff(d)
                
    current_scope = max(max(r) for r in current_rows)
    current_cost = current_scope + 1000 * current_conflict_count
    best_cost = current_cost
    best_rows_final = [list(r) for r in current_rows]
    best_scope_final = current_scope
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline - 5:
        if T < T_min:
            T = 2.0
            cooling = 0.9995
            if steps > 50000:
                break
                
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        p_idx = random.randint(1, len(row) - 1)
        
        old_val = row[p_idx]
        
        diffs_to_remove = []
        for i in range(len(row)):
            if i != p_idx:
                diffs_to_remove.append(abs(row[i] - old_val))
        
        for d in diffs_to_remove:
            remove_diff(d)
        
        row.pop(p_idx)
        
        new_val = None
        new_r_idx = r_idx
        
        if random.random() < 0.8:
            max_search = min(best_scope_final + 10, 250)
            found = False
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
                    new_val = cand
                    found = True
                    break
            
            if not found:
                pos = 0
                while pos < len(row) and row[pos] < old_val:
                    pos += 1
                row.insert(pos, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                continue
        else:
            new_r_idx = random.randint(0, n_rows - 1)
            while new_r_idx == r_idx:
                new_r_idx = random.randint(0, n_rows - 1)
            
            cand = old_val
            target_row = current_rows[new_r_idx]
            
            if cand in target_row:
                cand = random.randint(1, best_scope_final + 10)
                if cand in target_row:
                    pos = 0
                    while pos < len(row) and row[pos] < old_val:
                        pos += 1
                    row.insert(pos, old_val)
                    for d in diffs_to_remove:
                        add_diff(d)
                    continue
            
            valid = True
            new_diffs = []
            for a in target_row:
                d = abs(cand - a)
                if d in diff_counts:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                pos = 0
                while pos < len(target_row) and target_row[pos] < cand:
                    pos += 1
                target_row.insert(pos, cand)
                for d in new_diffs:
                    add_diff(d)
                new_val = cand
            else:
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
            
            if is_valid(current_rows):
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope_final = current_scope
                    best_rows_final = [list(r) for r in current_rows]
        else:
            if new_r_idx != r_idx:
                target_row = current_rows[new_r_idx]
                idx_new = target_row.index(new_val)
                target_row.pop(idx_new)
                for a in target_row:
                    remove_diff(abs(a - new_val))
            else:
                row = current_rows[r_idx]
                idx_new = row.index(new_val)
                row.pop(idx_new)
                for a in row:
                    remove_diff(abs(a - new_val))
            
            row = current_rows[r_idx]
            pos = 0
            while pos < len(row) and row[pos] < old_val:
                pos += 1
            row.insert(pos, old_val)
            for d in diffs_to_remove:
                add_diff(d)
        
        T *= cooling
        steps += 1

    # 6. Final Compaction
    final_rows = [list(r) for r in best_rows_final]
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
    
    if is_valid(final_rows) and final_scope < best_scope_final:
        best_rows_final = final_rows
        best_scope_final = final_scope
        
    if not is_valid(best_rows_final):
        if is_valid(best_rows):
            best_rows_final = best_rows
        else:
            used = set()
            best_rows_final = []
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
                best_rows_final.append(row)

    return best_rows_final
