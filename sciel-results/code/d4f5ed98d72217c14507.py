import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before the 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        """
        Counts the number of distinct difference collisions.
        """
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def greedy_init(rand_prob=0.1, window=20):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 300 
                while x <= limit and len(candidates) < window:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
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

    def local_search_targeted(rows, time_limit):
        """
        Structural refinement: Targeted descent to minimize the global maximum (scope).
        We repeatedly identify the largest element in the entire configuration.
        We attempt to replace it with a smaller valid value.
        If successful, the scope might decrease.
        If stuck, we perform random perturbations to escape local minima.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        current_conflicts = count_conflicts(best_rows)
        
        while time.time() < start + time_limit:
            # 1. Try to reduce the global maximum
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
                break # Can't change 0
            
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            
            # Build set of all diffs
            all_diffs = set()
            for r in best_rows:
                for i in range(row_len):
                    for j in range(i+1, row_len):
                        all_diffs.add(r[j] - r[i])
            
            # Diffs to remove (involving target)
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, row_len):
                remove_set.add(row[j] - current_val)
            
            fixed_diffs = all_diffs - remove_set
            
            found = False
            # Try values from current_val - 1 down to prev_val + 1
            # We want to find the smallest valid value to minimize scope, but reducing max is priority.
            # Actually, just finding ANY smaller valid value reduces the global max if it was the unique max.
            # But we want to minimize the final scope, so we should look for the smallest possible value
            # that is still valid? No, if we find a valid value smaller than current, we accept it.
            # To minimize scope, we should try to push it down as much as possible.
            
            # Let's iterate downwards. If we find a valid one, we accept it.
            # This might not be the absolute smallest, but it reduces the current bottleneck.
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                # Check diffs with left neighbors
                for i in range(max_c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Check diffs with right neighbors
                for j in range(max_c_idx + 1, row_len):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check for duplicates within new diffs
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, row_len)]
                
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                
                # Valid!
                row[max_c_idx] = new_val
                found = True
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                # Update conflicts? We changed the row, so conflicts might change.
                # But we ensured no conflicts with fixed_diffs.
                # However, we might have created conflicts within the row (checked above).
                # And we removed old diffs and added new ones.
                # Since new diffs are not in fixed_diffs, they are unique globally?
                # Yes, because fixed_diffs contains all other diffs.
                # So conflicts should not increase.
                # Actually, conflicts might decrease if the old value was causing a conflict
                # that is now removed.
                break
            
            if not found:
                # 2. If we can't reduce the current max, try a random swap on a non-max element to "jolt" the system.
                # Pick a random row and column (not 0)
                r_idx = random.randint(0, n_rows-1)
                c_idx = random.randint(1, row_len-1)
                
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx-1]
                next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 100
                
                # Re-compute fixed diffs for this position
                all_diffs = set()
                for r in best_rows:
                    for i in range(row_len):
                        for j in range(i+1, row_len):
                            all_diffs.add(r[j] - r[i])
                
                remove_set = set()
                row = best_rows[r_idx]
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, row_len):
                    remove_set.add(row[j] - curr_val)
                
                fixed_diffs = all_diffs - remove_set
                
                # Try a few random candidates
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len-1 else curr_val + 50
                
                if low > high:
                    continue
                
                # Try to find a valid random move
                for _ in range(50):
                    cand = random.randint(low, high)
                    if cand == curr_val:
                        continue
                    
                    valid = True
                    for i in range(c_idx):
                        d = cand - row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    for j in range(c_idx + 1, row_len):
                        d = row[j] - cand
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_left_diffs = [cand - row[i] for i in range(c_idx)]
                    new_right_diffs = [row[j] - cand for j in range(c_idx + 1, row_len)]
                    all_new = new_left_diffs + new_right_diffs
                    if len(set(all_new)) != len(all_new):
                        continue
                    
                    # Accept if it reduces scope or keeps it same
                    # We don't strictly need to check scope here, just accept to explore.
                    # But to be safe, let's accept if it doesn't increase conflicts.
                    # Since it's valid against fixed_diffs, it shouldn't create global conflicts.
                    row[c_idx] = cand
                    break
        
        return best_rows, best_scope

    def local_search_sa(rows, time_limit):
        """
        Simulated Annealing to escape local minima and refine scope.
        """
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        current_conflicts = count_conflicts(current_rows)
        
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        best_conflicts = current_conflicts
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        start = time.time()
        
        while time.time() < start + time_limit:
            if T < T_min:
                break
            
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            row = current_rows[r_idx]
            old_val = row[pos]
            
            lower_bound = row[pos - 1] + 1
            if pos < k:
                upper_bound = row[pos + 1] - 1
            else:
                upper_bound = current_scope + 10
            
            if lower_bound > upper_bound:
                continue
            
            # Pick a random new value
            v = random.randint(lower_bound, upper_bound)
            if v == old_val:
                continue
            
            row[pos] = v
            new_conflicts = count_conflicts(current_rows)
            new_scope = get_scope(current_rows)
            
            delta_conflict = new_conflicts - current_conflicts
            delta_scope = new_scope - current_scope
            
            accept = False
            if new_conflicts < current_conflicts:
                accept = True
            elif new_conflicts == current_conflicts:
                if new_scope < current_scope:
                    accept = True
                else:
                    if random.random() < math.exp(-delta_scope / T):
                        accept = True
            else:
                if random.random() < math.exp(-delta_conflict / T):
                    accept = True
            
            if accept:
                current_conflicts = new_conflicts
                current_scope = new_scope
                
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and current_scope < best_scope):
                    best_conflicts = current_conflicts
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                row[pos] = old_val
            
            T *= cooling
            
        return best_rows, best_scope

    # --- INSPIRED ADDITIONS: Row Reconstruction & ILS ---
    
    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def rebuild_row(used_other, jitter=0.0, max_val=None, rng_seed=None):
        """
        Rebuild a single row given differences already used by other rows.
        Returns the row list or None if failed.
        """
        if rng_seed is not None:
            local_rng = random.Random(rng_seed)
        else:
            local_rng = random

        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            search_end = max_val if max_val else (best_scope + 30)
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and local_rng.random() < jitter and trial_x < 300:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def local_polish(rows, max_passes=4):
        """
        Reduce scope by locally re-optimizing each row.
        """
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 10 and pass_count < max_passes:
            improved = False
            pass_count += 1
            # Process rows with largest max first
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 10:
                    break
                
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                
                # Try to rebuild row i with smaller max
                best_row = list(rows[i])
                best_max = max(rows[i])
                
                # Try greedy first (no jitter)
                new_row = rebuild_row(used_other, jitter=0.0, max_val=best_max - 1)
                if new_row is not None and max(new_row) < best_max:
                    best_row = new_row
                    best_max = max(new_row)
                
                # Try a few jittered rebuilds
                for attempt in range(10):
                    jitter = 0.1 + 0.05 * attempt
                    new_row = rebuild_row(used_other, jitter=jitter, max_val=best_max - 1)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        return rows

    def kick(rows, num_kicks=1):
        """
        Perturb by rebuilding random rows with high jitter.
        """
        rows = [list(r) for r in rows]
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            
            success = False
            for attempt in range(15):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row(used_other, jitter=jitter, max_val=best_scope + 20)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                # Fallback: greedy rebuild
                new_row = rebuild_row(used_other, jitter=0.0, max_val=best_scope + 30)
                if new_row is not None:
                    rows[idx] = new_row
                else:
                    # Keep original
                    pass
        return rows

    # --- MAIN SEARCH LOOP ---

    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')
    
    # Multi-start strategy
    # 1. Greedy init
    # 2. Targeted Descent (reduce max)
    # 3. SA (refine/escape)
    # 4. ILS with Row Reconstruction (new)
    
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Vary initialization
        rand_prob = 0.05 + 0.05 * i
        window = 10 + 5 * i
        
        rows = greedy_init(rand_prob, window)
        scope = get_scope(rows)
        conflicts = count_conflicts(rows)
        
        # Phase 1: Targeted Descent (fast, reduces scope significantly)
        # Allocate ~4s
        t1_rows, t1_scope = local_search_targeted(rows, 4.0)
        t1_conflicts = count_conflicts(t1_rows)
        
        if t1_conflicts < best_conflicts or (t1_conflicts == best_conflicts and t1_scope < best_scope):
            best_conflicts = t1_conflicts
            best_scope = t1_scope
            best_rows = t1_rows
            
        # Phase 2: SA (polish)
        # Allocate ~4s
        if time.time() > deadline - 15:
            break
            
        t2_rows, t2_scope = local_search_sa(t1_rows, 4.0)
        t2_conflicts = count_conflicts(t2_rows)
        
        if t2_conflicts < best_conflicts or (t2_conflicts == best_conflicts and t2_scope < best_scope):
            best_conflicts = t2_conflicts
            best_scope = t2_scope
            best_rows = t2_rows
            
        # Early exit if very good
        if best_conflicts == 0 and best_scope < 120:
            break

    # Phase 3: ILS with Row Reconstruction (Inspired)
    # This is a structural change: instead of tweaking individual elements, we rebuild whole rows.
    if best_rows is not None:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 10:
            # We will iterate: Kick -> Polish -> Check
            # Allocate time per iteration based on remaining time
            # Let's say each iteration takes ~2s (1s kick, 1s polish)
            num_il_iters = int(remaining_time / 2.5)
            num_il_iters = max(1, num_il_iters)
            
            for it in range(num_il_iters):
                if time.time() > deadline - 5:
                    break
                
                # Kick: Perturb current best
                cur = kick(best_rows, num_kicks=random.choice([1, 2]))
                
                # Validate
                cur_scope = get_scope(cur)
                cur_conflicts = count_conflicts(cur)
                
                # If kick broke validity (conflicts > 0), we might still try to polish it
                # But usually kick should produce valid rows because rebuild_row checks against used_other.
                # However, if rebuild fails, it keeps the old row, so validity is preserved.
                # Wait, if rebuild fails, it keeps the old row. So cur should be valid if best_rows was valid.
                
                # Polish: Rebuild rows to reduce scope
                polished = local_polish(cur, max_passes=2)
                pol_scope = get_scope(polished)
                pol_conflicts = count_conflicts(polished)
                
                if pol_conflicts < best_conflicts or (pol_conflicts == best_conflicts and pol_scope < best_scope):
                    best_conflicts = pol_conflicts
                    best_scope = pol_scope
                    best_rows = polished
                else:
                    # If polish didn't improve, maybe we should keep the kicked version if it's better than best?
                    # No, we only update best if improved.
                    # But we can use the kicked version for the next kick?
                    # ILS usually kicks from the *current* solution, not necessarily the *best*.
                    # But here we are kicking from best. If it doesn't improve, we stay at best.
                    # This is "Best-First" ILS. "First-Accepting" might escape local minima better.
                    # Let's try First-Accepting: if pol_conflicts == 0, accept it even if scope is slightly worse?
                    # No, we want to minimize scope.
                    # Let's stick to best-first for now to ensure progress.
                    pass

    # Final polish if time remains
    if best_rows is not None:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 5:
            # Alternate between Targeted and SA
            # Split time
            half_time = remaining_time / 2
            
            # Targeted
            f1_rows, f1_scope = local_search_targeted(best_rows, half_time)
            f1_conflicts = count_conflicts(f1_rows)
            
            if f1_conflicts < best_conflicts or (f1_conflicts == best_conflicts and f1_scope < best_scope):
                best_conflicts = f1_conflicts
                best_scope = f1_scope
                best_rows = f1_rows
            
            # SA
            if time.time() < deadline - 2.0:
                f2_rows, f2_scope = local_search_sa(best_rows, deadline - time.time() - 2.0)
                f2_conflicts = count_conflicts(f2_rows)
                
                if f2_conflicts < best_conflicts or (f2_conflicts == best_conflicts and f2_scope < best_scope):
                    best_conflicts = f2_conflicts
                    best_scope = f2_scope
                    best_rows = f2_rows

    if best_rows is None:
        best_rows = greedy_init(0.1, 20)
        
    return best_rows
