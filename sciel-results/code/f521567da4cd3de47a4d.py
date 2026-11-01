import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def get_all_diffs(rows):
        s = set()
        for r in rows:
            s |= row_diffs_set(r)
        return s

    def greedy_init(rand_prob=0.1, window=20, max_limit=300):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                candidates = []
                x = last + 1
                # Limit search window for speed, expand if stuck
                limit = max_limit
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # If no candidates in window, search further
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
            rows.append(row)
        return rows

    def rebuild_row(used_other, max_val=None, jitter=0.0):
        """
        Reconstruct a single row given the diffs used by all other rows.
        Returns None if impossible within max_val (if provided) or reasonable bounds.
        """
        row = [0]
        row_diffs = set()
        
        # Estimate a reasonable upper bound if not provided
        if max_val is None:
             max_val = 200 + int(jitter * 100)

        while len(row) < row_len:
            last = row[-1]
            found = False
            # Search from last+1 up to max_val
            for trial_x in range(last + 1, max_val + 1):
                # Check against own diffs
                own_new_diffs = {trial_x - a for a in row}
                if own_new_diffs & row_diffs:
                    continue
                # Check against other rows' diffs
                if own_new_diffs & used_other:
                    continue
                
                # Apply jitter: skip some valid early candidates to diversify
                if jitter > 0 and random.random() < jitter:
                    continue

                row.append(trial_x)
                row_diffs |= own_new_diffs
                found = True
                break
            
            if not found:
                return None
        
        return row

    def targeted_descent(rows, time_limit):
        """
        Try to reduce the scope by moving elements down.
        Focuses on the global maximum first, then other large elements.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        while time.time() < start + time_limit:
            # Find global max
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                break # Can't reduce 0
                
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            
            # Get all diffs
            all_diffs = get_all_diffs(best_rows)
            
            # Diffs to remove (involving target)
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, row_len):
                remove_set.add(row[j] - current_val)
            
            fixed_diffs = all_diffs - remove_set
            
            found = False
            # Try to decrease
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                for i in range(max_c_idx):
                    if (new_val - row[i]) in fixed_diffs:
                        valid = False
                        break
                if not valid: continue
                for j in range(max_c_idx + 1, row_len):
                    if (row[j] - new_val) in fixed_diffs:
                        valid = False
                        break
                if not valid: continue
                
                # Check internal consistency of new diffs
                new_left = [new_val - row[i] for i in range(max_c_idx)]
                new_right = [row[j] - new_val for j in range(max_c_idx + 1, row_len)]
                all_new = new_left + new_right
                if len(set(all_new)) != len(all_new):
                    continue
                
                row[max_c_idx] = new_val
                found = True
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                break
            
            if not found:
                # If global max can't be reduced, try to reduce other large elements
                # Or break out to allow perturbation
                # For efficiency in ILS, we break to let the main loop perturb
                break
                
        return best_rows, best_scope

    def high_precision_polish(rows, time_limit):
        """
        Exhaustively try to reduce every non-first element in each row.
        This is the most effective local optimizer for this problem.
        """
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        pass_count = 0
        max_passes = 50 # Allow many passes
        
        while improved and pass_count < max_passes and (time.time() - start) < time_limit:
            improved = False
            pass_count += 1
            # Process rows in order of largest max value first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            
            for r_idx in order:
                if (time.time() - start) > time_limit:
                    break
                row = rows[r_idx]
                
                # Diffs from all other rows
                other_diffs = set()
                for j in range(n_rows):
                    if j != r_idx:
                        other_diffs |= row_diffs_set(rows[j])
                
                # Try to reduce elements from right to left (largest to smallest)
                for c_idx in range(row_len - 1, 0, -1):
                    if (time.time() - start) > time_limit:
                        break
                    curr_val = row[c_idx]
                    prev_val = row[c_idx - 1]
                    
                    # Diffs involving this element in the current row
                    remove_set = set()
                    for i in range(c_idx):
                        remove_set.add(curr_val - row[i])
                    for j in range(c_idx + 1, row_len):
                        remove_set.add(row[j] - curr_val)
                    
                    # Fixed diffs: all diffs from other rows + diffs from same row not involving c_idx
                    # Note: We must ensure new diffs don't clash with existing fixed diffs
                    # Also, new diffs must be distinct from each other
                    fixed_diffs = other_diffs.copy()
                    
                    # Add same-row diffs not involving c_idx
                    for i in range(row_len):
                        if i == c_idx: continue
                        for j in range(i + 1, row_len):
                            if j == c_idx: continue
                            fixed_diffs.add(row[j] - row[i])
                    
                    # Try to decrease
                    found = False
                    for new_val in range(curr_val - 1, prev_val, -1):
                        valid = True
                        new_diffs = set()
                        
                        # Check left neighbors
                        for i in range(c_idx):
                            d = new_val - row[i]
                            if d in fixed_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if not valid: continue
                        
                        # Check right neighbors
                        for j in range(c_idx + 1, row_len):
                            d = row[j] - new_val
                            if d in fixed_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if not valid: continue
                        
                        # If we reach here, it's valid
                        row[c_idx] = new_val
                        found = True
                        improved = True
                        break
                    
                    if found:
                        # If we changed the row, the 'other_diffs' for subsequent elements in this row
                        # are technically still valid because we only moved one element, 
                        # but the 'fixed_diffs' calculation for the NEXT element (c_idx-1) 
                        # needs to account for the new value. 
                        # However, since we are iterating right-to-left, and we just changed c_idx,
                        # the diffs involving c_idx are "active" now. 
                        # When we process c_idx-1, we will recalculate remove_set and fixed_diffs.
                        # So we don't need to update other_diffs immediately, just proceed.
                        pass

        return rows

    def kick(rows, num_kicks=1):
        """
        Perturb the solution by reconstructing one or more rows.
        """
        rows = [list(r) for r in rows]
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            success = False
            # Try to rebuild with increasing randomness (jitter)
            for attempt in range(10):
                jitter = 0.1 + 0.1 * attempt
                # Allow a slightly larger max value to ensure success, polish will reduce it
                max_val = get_scope(rows) + 20 
                new_row = rebuild_row(used_other, max_val=max_val, jitter=jitter)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            
            if not success:
                # Fallback: try without jitter, larger bound
                new_row = rebuild_row(used_other, max_val=get_scope(rows) + 50, jitter=0.0)
                if new_row is not None:
                    rows[idx] = new_row
                else:
                    # If even that fails, keep old row (should be rare)
                    pass
        
        return rows

    # --- MAIN SEARCH LOOP ---
    best_rows = None
    best_scope = float('inf')
    
    # 1. Initial Phase: Generate diverse starts and polish them
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 20:
            break
        
        # Vary initialization parameters
        rand_prob = 0.05 + 0.05 * i
        window = 10 + 5 * i
        
        rows = greedy_init(rand_prob, window)
        
        # Quick polish to get a decent baseline
        polished = high_precision_polish(rows, time_limit=2.0)
        scope = get_scope(polished)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = polished
            
        # Early exit if we hit the theoretical lower bound or close to best known
        if best_scope <= 112:
            break

    if best_rows is None:
        best_rows = greedy_init(0.1, 20)
        best_scope = get_scope(best_rows)
        best_rows = high_precision_polish(best_rows, time_limit=5.0)
        best_scope = get_scope(best_rows)

    # 2. ILS Phase: Iterate Descent -> Kick -> Polish
    current_rows = best_rows
    current_scope = best_scope
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline - 10:
        # Step 1: Targeted Descent (fast local improvement)
        descended_rows, descended_scope = targeted_descent(current_rows, time_limit=0.5)
        
        # Step 2: High Precision Polish (thorough local improvement)
        polished_rows = high_precision_polish(descended_rows, time_limit=1.0)
        polished_scope = get_scope(polished_rows)
        
        # Check for improvement
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_rows = polished_rows
            current_rows = polished_rows
            current_scope = polished_scope
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Step 3: Kick (Perturbation)
        if no_improve_count > max_no_improve or random.random() < 0.3:
            # If stuck, kick harder
            num_kicks = 1 if no_improve_count < max_no_improve * 2 else 2
            current_rows = kick(best_rows, num_kicks=num_kicks)
            current_scope = get_scope(current_rows)
            no_improve_count = 0
            
        # Safety check
        if time.time() > deadline - 10:
            break

    # 3. Final Polish: Use remaining time to polish the best solution found
    if best_rows is not None:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 1.0:
            final_polished = high_precision_polish(best_rows, time_limit=remaining_time)
            final_scope = get_scope(final_polished)
            if final_scope < best_scope:
                best_rows = final_polished
                best_scope = final_scope

    return best_rows
