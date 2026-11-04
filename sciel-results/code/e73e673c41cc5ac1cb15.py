import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    # k is the number of intervals, so row length is k+1
    # Task: "Choose 4 rows of 8 increasing integers starting at 0"
    # "Each row is a list of k + 1 integers"
    
    def get_row_bits(row):
        bits = 0
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                bits |= (1 << (row[j] - row[i]))
        return bits

    def count_bits(x):
        return bin(x).count('1')

    def is_valid_move(row, j, new_val, row_bits, global_bits):
        # Check bounds
        if j > 0 and new_val <= row[j-1]:
            return False, 0
        if j < len(row) - 1 and new_val >= row[j+1]:
            return False, 0
        
        # Calculate new row bits
        # Optimization: only recalculate diffs involving j? 
        # No, just recalculate all, it's fast enough for k=8
        new_rb = 0
        n = len(row)
        for i in range(n):
            v_i = row[i] if i != j else new_val
            for l in range(i+1, n):
                v_l = row[l] if l != j else new_val
                d = v_l - v_i
                new_rb |= (1 << d)
        
        # Check internal distinctness
        expected_pairs = n * (n - 1) // 2
        if count_bits(new_rb) != expected_pairs:
            return False, 0
            
        # Check conflict with other rows
        other_bits = global_bits ^ row_bits
        if (new_rb & other_bits) != 0:
            return False, 0
        
        return True, new_rb

    def greedy_randomized(rows_count, k_val, rng):
        used_bits = 0
        rows = []
        for _ in range(rows_count):
            row = [0]
            x = 0
            while len(row) < k_val + 1:
                candidates = []
                # Look ahead
                for offset in range(20):
                    trial_x = x + offset
                    if trial_x < 0: continue
                    new_diffs = {trial_x - a for a in row}
                    nd_bits = 0
                    for d in new_diffs:
                        nd_bits |= (1 << d)
                    
                    # Internal distinctness is guaranteed for new element at end
                    # Check global conflict
                    if (nd_bits & used_bits) == 0:
                        candidates.append((trial_x, nd_bits))
                        if len(candidates) >= 5:
                            break
                
                if not candidates:
                    # Try wider range
                    for offset in range(20, 100):
                        trial_x = x + offset
                        new_diffs = {trial_x - a for a in row}
                        nd_bits = 0
                        for d in new_diffs:
                            nd_bits |= (1 << d)
                        if (nd_bits & used_bits) == 0:
                            candidates.append((trial_x, nd_bits))
                            break
                    if not candidates:
                        return None
                            
                chosen_x, chosen_bits = rng.choice(candidates)
                row.append(chosen_x)
                used_bits |= chosen_bits
                x = chosen_x
            rows.append(row)
        return rows

    def local_search(rows, rng, deadline):
        row_bits_list = [get_row_bits(r) for r in rows]
        global_bits = 0
        for rb in row_bits_list:
            global_bits |= rb
        
        current_scope = max(max(r) for r in rows)
        best_scope = current_scope
        best_rows = [r[:] for r in rows]
        
        no_improve_count = 0
        max_no_improve = 5000
        expected_pairs = len(rows[0]) * (len(rows[0]) - 1) // 2

        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break
            
            r_idx = rng.randrange(len(rows))
            j_idx = rng.randrange(1, len(rows[r_idx]))
            
            row = rows[r_idx]
            old_val = row[j_idx]
            
            lower = row[j_idx-1] + 1
            upper = row[j_idx+1] - 1 if j_idx < len(row)-1 else 1000
            
            # Try to reduce value
            found_improvement = False
            for _ in range(5):
                if lower > old_val:
                    break
                new_val = rng.randrange(lower, old_val + 1)
                
                valid, new_rb = is_valid_move(row, j_idx, new_val, row_bits_list[r_idx], global_bits)
                if valid:
                    # Calculate new scope
                    new_row = row[:]
                    new_row[j_idx] = new_val
                    new_row_max = max(new_row)
                    max_other = 0
                    for i, rr in enumerate(rows):
                        if i == r_idx: continue
                        if max(rr) > max_other:
                            max_other = max(rr)
                    new_global_scope = max(new_row_max, max_other)
                    
                    if new_global_scope <= current_scope:
                        # Update state
                        old_rb = row_bits_list[r_idx]
                        global_bits = (global_bits ^ old_rb) | new_rb
                        row_bits_list[r_idx] = new_rb
                        rows[r_idx] = new_row
                        current_scope = new_global_scope
                        
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        found_improvement = True
                        break
            if not found_improvement:
                no_improve_count += 1
                
        return best_rows, best_scope

    best_rows = None
    best_scope = float('inf')
    
    # Initial deterministic greedy
    rows = greedy_randomized(n_rows, k, random.Random(0))
    if rows:
        best_rows, best_scope = local_search(rows, random.Random(1), deadline)
    
    # Randomized restarts
    seed = 10
    while time.time() < deadline - 5.0:
        rng = random.Random(seed)
        rows = greedy_randomized(n_rows, k, rng)
        if rows is None:
            seed += 1
            continue
            
        ls_rng = random.Random(seed * 7 + 1)
        res_rows, res_scope = local_search(rows, ls_rng, deadline)
        
        if res_scope < best_scope:
            best_scope = res_scope
            best_rows = res_rows
            
        seed += 1
        
    if best_rows is None:
        # Fallback
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
            
    return best_rows
