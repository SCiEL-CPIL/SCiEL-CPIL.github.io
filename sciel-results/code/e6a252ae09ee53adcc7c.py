import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 132.0  # Safety margin before 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init_interleaved(seed=0):
        """
        Interleaved greedy: builds all rows in parallel, always extending
        the row with the fewest elements (tie-break: smallest last value).
        This promotes balanced row lengths and often tighter packing.
        """
        random.seed(seed)
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        # Total elements to add: n_rows * k
        for _ in range(n_rows * k):
            # Find the row to extend
            # Heuristic: min length, then min last value
            best_idx = -1
            best_key = None
            
            for i in range(n_rows):
                if len(rows[i]) < row_len:
                    # Key: (length, last_val, random_noise for diversity)
                    noise = random.random() * 0.1
                    key = (len(rows[i]), rows[i][-1] + noise)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = i
            
            if best_idx == -1:
                break # Should not happen if logic is correct
                
            row = rows[best_idx]
            last = row[-1]
            
            # Find the smallest valid next value
            x = last + 1
            while True:
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used_diffs.add(x - a)
                    break
                x += 1
        
        return rows

    def local_search(rows, time_limit):
        """
        Structural refinement: Systematic scope descent with full sweeps.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        improved = True
        
        while time.time() < start + time_limit and improved:
            improved = False
            
            # Full sweep: process elements in descending order of value
            elements = []
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    elements.append((best_rows[r_idx][c_idx], r_idx, c_idx))
            elements.sort(reverse=True)
            
            for val, r_idx, c_idx in elements:
                if time.time() > start + time_limit:
                    break
                
                row = best_rows[r_idx]
                current_val = row[c_idx]
                prev_val = row[c_idx - 1]
                
                if current_val <= prev_val:
                    continue
                
                # Compute all diffs, excluding those involving this element
                all_diffs = set()
                for r in best_rows:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            all_diffs.add(r[j] - r[i])
                
                remove_set = set()
                for i in range(c_idx):
                    remove_set.add(current_val - row[i])
                for j in range(c_idx + 1, len(row)):
                    remove_set.add(row[j] - current_val)
                
                fixed_diffs = all_diffs - remove_set
                
                # Try to pull this element down as far as possible
                found = False
                for new_val in range(current_val - 1, prev_val, -1):
                    valid = True
                    for i in range(c_idx):
                        d = new_val - row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    for j in range(c_idx + 1, len(row)):
                        d = row[j] - new_val
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_left_diffs = [new_val - row[i] for i in range(c_idx)]
                    new_right_diffs = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                    
                    all_new = new_left_diffs + new_right_diffs
                    if len(set(all_new)) != len(all_new):
                        continue
                    
                    row[c_idx] = new_val
                    found = True
                    improved = True
                    break
                
                # If we couldn't pull down, try a random perturbation to escape local minima
                if not found and time.time() < start + time_limit:
                    low = prev_val + 1
                    next_v = row[c_idx + 1] if c_idx < row_len - 1 else current_val + 50
                    high = next_v - 1 if c_idx < row_len - 1 else current_val + 50
                    
                    if low <= high:
                        for _ in range(10):
                            if time.time() > start + time_limit:
                                break
                            cand = random.randint(low, high)
                            if cand == current_val:
                                continue
                            
                            valid = True
                            for i in range(c_idx):
                                d = cand - row[i]
                                if d in fixed_diffs:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            for j in range(c_idx + 1, len(row)):
                                d = row[j] - cand
                                if d in fixed_diffs:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            new_left = [cand - row[i] for i in range(c_idx)]
                            new_right = [row[j] - cand for j in range(c_idx + 1, len(row))]
                            if len(set(new_left + new_right)) != len(new_left + new_right):
                                continue
                            
                            row[c_idx] = cand
                            improved = True
                            break
        
        # Final polish: exhaustive single-element optimization
        for _ in range(3):
            if time.time() > start + time_limit:
                break
            for r_idx in range(n_rows):
                for c_idx in range(1, row_len):
                    if time.time() > start + time_limit:
                        break
                    row = best_rows[r_idx]
                    current_val = row[c_idx]
                    prev_val = row[c_idx - 1]
                    
                    all_diffs = set()
                    for r in best_rows:
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                all_diffs.add(r[j] - r[i])
                    
                    remove_set = set()
                    for i in range(c_idx):
                        remove_set.add(current_val - row[i])
                    for j in range(c_idx + 1, len(row)):
                        remove_set.add(row[j] - current_val)
                    
                    fixed_diffs = all_diffs - remove_set
                    
                    for new_val in range(current_val - 1, prev_val, -1):
                        valid = True
                        for i in range(c_idx):
                            d = new_val - row[i]
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        for j in range(c_idx + 1, len(row)):
                            d = row[j] - new_val
                            if d in fixed_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        new_left = [new_val - row[i] for i in range(c_idx)]
                        new_right = [row[j] - new_val for j in range(c_idx + 1, len(row))]
                        if len(set(new_left + new_right)) != len(new_left + new_right):
                            continue
                        row[c_idx] = new_val
                        break
        
        best_scope = get_scope(best_rows)
        return best_rows, best_scope

    def beam_search(rows_init, time_limit, best_scope_bound):
        """
        Lightweight beam search to explore neighborhood.
        State: (rows, used_diffs)
        """
        start = time.time()
        beam_width = 50
        beam = [(rows_init, set())]
        
        # Pre-compute used diffs for initial state
        used_diffs = set()
        for r in rows_init:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    used_diffs.add(r[j] - r[i])
        beam = [(rows_init, used_diffs)]
        
        best_found = [r[:] for r in rows_init]
        best_cost = best_scope_bound
        
        while time.time() < start + time_limit:
            next_beam = []
            for rows, used in beam:
                # If complete, check cost
                if all(len(r) == row_len for r in rows):
                    cost = get_scope(rows)
                    if cost < best_cost:
                        best_cost = cost
                        best_found = [r[:] for r in rows]
                    continue
                
                # Identify candidate moves: pick a row to extend
                # Heuristic: row with min length
                min_len = row_len
                target_row_idx = -1
                min_last_val = float('inf')
                
                for i in range(n_rows):
                    if len(rows[i]) < min_len:
                        min_len = len(rows[i])
                        target_row_idx = i
                        min_last_val = rows[i][-1]
                    elif len(rows[i]) == min_len:
                        if rows[i][-1] < min_last_val:
                            min_last_val = rows[i][-1]
                            target_row_idx = i
                
                if target_row_idx == -1:
                    continue
                    
                row = rows[target_row_idx]
                last = row[-1]
                
                # Generate candidates for this row
                candidates = []
                x = last + 1
                # Limit search space to best_cost - 1
                limit = best_cost - 1
                
                while x <= limit and len(candidates) < 10:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                for val in candidates:
                    new_rows = [r[:] for r in rows]
                    new_rows[target_row_idx].append(val)
                    
                    new_used = set(used)
                    for a in new_rows[target_row_idx]:
                        if a != val:
                            new_used.add(val - a)
                    
                    new_cost = get_scope(new_rows)
                    if new_cost < best_cost:
                        next_beam.append((new_rows, new_used))
            
            if not next_beam:
                break
                
            # Sort by cost
            next_beam.sort(key=lambda x: get_scope(x[0]))
            
            # Keep top beam_width, with some diversity
            # Simple: take top beam_width
            beam = next_beam[:beam_width]
            
            # Early stop if we found a perfect solution (unlikely but check)
            if best_cost <= 112: 
                break
                
        return best_found, best_cost

    # Phase 1: Get a baseline with interleaved greedy (multiple seeds)
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(5):
        if time.time() > deadline - 100:
            break
        rows = greedy_init_interleaved(seed)
        scope = get_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    if best_rows is None:
        # Fallback to simple greedy if interleaved failed or timed out
        rows = greedy_init_interleaved(0)
        best_rows = rows
        best_scope = get_scope(rows)

    # Phase 2: Local Search Refinement
    local_time = 40.0
    elapsed = time.time() - start_time
    if elapsed > deadline - 80:
        local_time = max(10, deadline - elapsed - 40)
    
    if local_time > 5:
        refined_rows, refined_scope = local_search(best_rows, local_time)
        if validate(refined_rows):
            best_rows = refined_rows
            best_scope = refined_scope

    # Phase 3: Beam Search
    beam_time = 30.0
    elapsed = time.time() - start_time
    if elapsed > deadline - 50:
        beam_time = max(5, deadline - elapsed - 20)
        
    if beam_time > 5:
        beam_rows, beam_scope = beam_search(best_rows, beam_time, best_scope)
        if validate(beam_rows):
            if beam_scope < best_scope:
                best_rows = beam_rows
                best_scope = beam_scope

    # Phase 4: Final Local Search Polish
    if time.time() < deadline - 10:
        polish_time = min(10.0, deadline - time.time() - 5)
        if polish_time > 2:
            polished_rows, polished_scope = local_search(best_rows, polish_time)
            if validate(polished_rows):
                best_rows = polished_rows
                best_scope = polished_scope

    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
