import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    
    # Safety margin: The evaluator kills at 150s. We aim to finish by 115s to be safe.
    deadline = _START + 115.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        """Check if rows are valid (strictly increasing, start at 0, unique diffs)."""
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(1, len(r)):
                if r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_init(rand_prob=0.1, window=20, seed=None):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        if seed is not None:
            random.seed(seed)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 200 
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
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

    def rebuild_row(used_other, jitter=0.0, max_val=None, rng_seed=None):
        """Rebuild a single row given differences already used by other rows."""
        if rng_seed is not None:
            local_rng = random.Random(rng_seed)
        else:
            local_rng = random.Random()

        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            search_end = max_val if max_val else (1000)
            # Ensure search_end is at least x+1
            if search_end <= x:
                search_end = x + 100
                
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

    def targeted_descent(rows, time_budget):
        """
        Aggressively try to reduce the scope by targeting the largest elements.
        Optimized for speed: O(k) validity check per candidate value.
        """
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        
        # Pre-compute all differences to allow O(1) membership checks
        # We need to handle the "remove and check" logic carefully.
        
        def get_all_diffs(r_idx, c_idx):
            """Get set of all diffs EXCLUDING those involving cur[r_idx][c_idx]."""
            diffs = set()
            row = cur[r_idx]
            val = row[c_idx]
            
            # Diffs within the same row that do NOT involve c_idx
            for i in range(len(row)):
                if i == c_idx: continue
                for j in range(i+1, len(row)):
                    if j == c_idx: continue
                    diffs.add(row[j] - row[i])
            
            # Diffs from all other rows
            for i in range(n_rows):
                if i == r_idx: continue
                r = cur[i]
                for a in range(len(r)):
                    for b in range(a+1, len(r)):
                        diffs.add(r[b] - r[a])
            return diffs

        # Main loop
        no_improve_count = 0
        while time.time() < start + time_budget:
            # Find global max
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(cur):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                break # Cannot reduce 0
            
            current_val = max_val
            prev_val = cur[max_r_idx][max_c_idx - 1]
            row = cur[max_r_idx]
            
            # Get fixed diffs (O(N^2) but N is small, ~100 ops)
            fixed_diffs = get_all_diffs(max_r_idx, max_c_idx)
            
            found_valid = False
            # Try to reduce current_val
            # We try values in [prev_val+1, current_val-1]
            # Heuristic: Try near current_val first (to keep row tight) or near prev_val?
            # Usually reducing the max is the goal. Any reduction helps.
            # Let's try a window of 20 values downwards from current_val.
            
            best_nv = None
            for step in range(20):
                nv = current_val - step
                if nv <= prev_val:
                    break
                
                # Check if nv creates any conflict
                # Diffs with left neighbors: nv - row[i]
                # Diffs with right neighbors: row[j] - nv
                valid = True
                new_diffs = set()
                
                for i in range(max_c_idx):
                    d = nv - row[i]
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                for j in range(max_c_idx + 1, len(row)):
                    d = row[j] - nv
                    if d in fixed_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    best_nv = nv
                    break
            
            if best_nv is not None:
                row[max_c_idx] = best_nv
                new_scope = get_scope(cur)
                if new_scope < cur_scope:
                    cur_scope = new_scope
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # If we didn't improve scope, we might be stuck.
                    # But we did make a move. Let's continue.
                
                # Time check
                if time.time() > start + time_budget:
                    break
            else:
                # Try random non-max elements to escape
                if no_improve_count > 5:
                    break
                
                # Pick a random element > 0
                r_idx = random.randint(0, n_rows-1)
                c_idx = random.randint(1, row_len-1)
                
                if r_idx == max_r_idx and c_idx == max_c_idx:
                    continue
                
                current_val = cur[r_idx][c_idx]
                prev_val = cur[r_idx][c_idx - 1]
                row = cur[r_idx]
                fixed_diffs = get_all_diffs(r_idx, c_idx)
                
                best_nv = None
                for step in range(10):
                    nv = current_val - step
                    if nv <= prev_val:
                        break
                    
                    valid = True
                    new_diffs = set()
                    
                    for i in range(c_idx):
                        d = nv - row[i]
                        if d in fixed_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    for j in range(c_idx + 1, len(row)):
                        d = row[j] - nv
                        if d in fixed_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        best_nv = nv
                        break
                
                if best_nv is not None:
                    row[c_idx] = best_nv
                    new_scope = get_scope(cur)
                    if new_scope < cur_scope:
                        cur_scope = new_scope
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

        return cur, cur_scope

    def kick(rows, num_kicks=1):
        """Perturb by rebuilding random rows with high jitter."""
        rows = [list(r) for r in rows]
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    row = rows[j]
                    for i in range(len(row)):
                        for l in range(i+1, len(row)):
                            used_other.add(row[l] - row[i])
            
            success = False
            for attempt in range(10):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row(used_other, jitter=jitter, max_val=get_scope(rows) + 20)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                # Fallback: greedy rebuild
                new_row = rebuild_row(used_other, jitter=0.0, max_val=get_scope(rows) + 30)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    # Phase 1: Initialization
    best_rows = greedy_init(rand_prob=0.1, seed=42)
    best_scope = get_scope(best_rows)
    
    # Phase 2: ILS with Targeted Descent
    max_iters = 1000
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break

        if it == 0:
            cur = [r[:] for r in best_rows]
        else:
            # Kick every few iterations to escape local minima
            if it % 5 == 0:
                cur = kick(best_rows, num_kicks=random.choice([1, 1, 2]))
            else:
                cur = [r[:] for r in best_rows]
                
        # Ensure validity after kick (rebuild_row should guarantee it, but check)
        if not validate(cur):
            continue

        cur_scope = get_scope(cur)
        
        # Apply Targeted Descent
        # Give it a tight budget of 0.5 seconds per iteration to allow many iterations
        desc_rows, desc_scope = targeted_descent(cur, time_budget=0.5)
        
        if desc_rows is not None:
            if desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows
            
    # Phase 3: Final Polish
    # Try to squeeze out the last bit by systematically reducing the last element of each row
    if time.time() < deadline - 2:
        cur = [r[:] for r in best_rows]
        cur_scope = get_scope(cur)
        
        # Systematic reduction of last elements
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for r_idx in range(n_rows):
                row = cur[r_idx]
                c_idx = len(row) - 1
                current_val = row[c_idx]
                prev_val = row[c_idx - 1]
                
                # Try to reduce
                # Use the optimized check from targeted_descent logic
                fixed_diffs = set()
                # Calculate fixed diffs for this specific element
                for i in range(n_rows):
                    if i == r_idx:
                        continue
                    r = cur[i]
                    for a in range(len(r)):
                        for b in range(a+1, len(r)):
                            fixed_diffs.add(r[b] - r[a])
                
                for a in range(len(row)):
                    if a == c_idx:
                        continue
                    for b in range(a+1, len(row)):
                        if b == c_idx:
                            continue
                        fixed_diffs.add(row[b] - row[a])
                
                # Try a few values
                for nv in range(current_val - 1, max(prev_val, current_val - 50), -1):
                    valid = True
                    new_diffs = set()
                    for i in range(c_idx):
                        d = nv - row[i]
                        if d in fixed_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    if valid:
                        row[c_idx] = nv
                        new_scope = get_scope(cur)
                        if new_scope < cur_scope:
                            cur_scope = new_scope
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [r[:] for r in cur]
                        improved = True
                        break # Restart loop with new state? Or just continue?
                        # Breaking inner loop, but 'improved' is True so outer while continues.
                        # But we modified row, so we should probably restart the scan of rows
                        # or at least this row.
                        # Let's just break and let the while loop restart.
                        break
        
        # One last targeted descent on the best
        if time.time() < deadline - 0.5:
            desc_rows, desc_scope = targeted_descent(best_rows, time_budget=0.5)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

    return best_rows
