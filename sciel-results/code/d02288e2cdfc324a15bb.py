import time
import random

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 115s to be safe.
    deadline = start_time + 115 

    # Precompute number of differences per row
    n_diffs_per_row = k * (k + 1) // 2

    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_global(rows):
        all_diffs = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def used_from_rows(rows):
        used = set()
        for r in rows:
            used |= get_row_diffs(r)
        return used

    def max_val(rows):
        return max(max(r) for r in rows)

    def greedy_row(used, mode=0, temp=0.0):
        """
        Build one row greedily.
        mode=0: Standard greedy (smallest valid)
        mode=1: Spread greedy (skip small valids to reserve them, or pick larger)
        mode=2: Randomized window (pick random from next few valids)
        """
        row = [0]
        local_diffs = set()
        x = 0
        
        while len(row) < k + 1:
            found = False
            # Increment x. In standard greedy, we check x+1, x+2...
            # In spread, we might jump.
            # Let's stick to a unified loop but change selection logic.
            
            # We need to find the next element.
            # We scan candidates starting from row[-1] + 1.
            last_val = row[-1]
            candidates = []
            
            # Limit search window to avoid infinite loops if stuck (shouldn't happen usually)
            search_limit = last_val + 500 
            
            for v in range(last_val + 1, search_limit):
                new_diffs = {v - a for a in row}
                
                # Check against global used and local diffs
                if new_diffs & used:
                    continue
                if new_diffs & local_diffs:
                    continue
                
                # Valid candidate found
                candidates.append(v)
                
                # Selection Logic based on mode
                if mode == 0:
                    # Standard Greedy: Take the first valid one
                    row.append(v)
                    local_diffs |= new_diffs
                    used |= new_diffs
                    found = True
                    break
                elif mode == 1:
                    # Spread Greedy: Take a valid one further out
                    # We collect a few and pick the largest in a small window
                    # to "spread" the row
                    if len(candidates) >= 3:
                        # Pick the 3rd valid one (or random from last few)
                        v_pick = candidates[-1]
                        row.append(v_pick)
                        local_diffs |= {v_pick - a for a in row[:-1]}
                        used |= local_diffs # Note: used is updated outside? No, we update local and global at end of row?
                        # Wait, the original code updates `used` inside the loop.
                        # We must be careful. `used` is passed by reference (set).
                        # We should only commit to `used` when the whole row is built?
                        # No, the original code does `used |= new_diffs` inside the loop.
                        # This is correct for sequential construction.
                        
                        # However, for `mode=1`, we are skipping `candidates[0]` and `candidates[1]`.
                        # Those differences are NOT used by this row, so they remain available for other rows.
                        # This is the key "reservation" strategy.
                        
                        # Update local_diffs and used with the picked v
                        new_diffs_picked = {v_pick - a for a in row[:-1]}
                        local_diffs |= new_diffs_picked
                        used |= new_diffs_picked
                        found = True
                        break
                elif mode == 2:
                    # Randomized: Collect up to 5 valids, pick one randomly
                    if len(candidates) >= 5:
                        v_pick = random.choice(candidates)
                        new_diffs_picked = {v_pick - a for a in row[:-1]}
                        row.append(v_pick)
                        local_diffs |= new_diffs_picked
                        used |= new_diffs_picked
                        found = True
                        break
                    # If we haven't collected enough, keep scanning
                    # But if we reach the end of the search limit, we should just take the last one?
                    # For now, if we exit the loop without found, we need a fallback.

            if not found:
                # Fallback: If we didn't find a candidate in the window (unlikely for valid sets),
                # extend the search or just take the next valid one greedily from where we left off.
                # This is a safety net.
                v = last_val + 1
                while True:
                    new_diffs = {v - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & local_diffs):
                        row.append(v)
                        local_diffs |= new_diffs
                        used |= new_diffs
                        break
                    v += 1

        return row, used

    def find_replacement(fixed_used, current_row, max_budget):
        """Try many candidate rows; return the best (lowest max) that fits."""
        best_local = None
        best_local_max = max(current_row) if current_row else float('inf')
        
        # Use diverse modes for replacement too
        modes = [0, 1, 2]
        temps = [0.0, 0.1] # Temp is less relevant now that modes handle diversity, but keep for randomization in mode 2
        
        for mode in modes:
            if time.time() > deadline - 5:
                break
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                # Copy fixed_used because greedy_row mutates it
                trial_used = set(fixed_used)
                trial_row, _ = greedy_row(trial_used, mode=mode)
                
                # Check if the generated row is valid against fixed_used (should be by construction)
                # But check if it's better
                if max(trial_row) < best_local_max:
                    # Verify global validity just in case of bugs in greedy logic
                    # Actually, if greedy_row is correct, it's valid against fixed_used.
                    # We just need to ensure it doesn't conflict with itself (handled by local_diffs)
                    # and fixed_used (handled by used check).
                    best_local = trial_row
                    best_local_max = max(trial_row)
        return best_local

    # ---------- Phase 1: Diverse Initial Construction ----------
    best_rows = None
    best_max = float('inf')

    # We will run several full constructions with different modes
    # Mode 0: All rows compact
    # Mode 1: All rows spread
    # Mode 2: Mixed modes (random per row)
    
    strategies = [
        [0] * n_rows,  # All compact
        [1] * n_rows,  # All spread
        [2] * n_rows,  # All randomized
        [0, 1, 2, 0, 1, 2][:n_rows] # Mixed pattern
    ]
    
    # Also add some random mix strategies
    for _ in range(5):
        strategies.append([random.choice([0, 1, 2]) for _ in range(n_rows)])

    for strat in strategies:
        if time.time() > deadline - 40:
            break
            
        used = set()
        trial_rows = []
        success = True
        
        for i in range(n_rows):
            if time.time() > deadline - 40:
                success = False
                break
            mode = strat[i]
            row, used = greedy_row(used, mode=mode)
            trial_rows.append(row)
            
        if not success:
            continue
            
        if is_valid_global(trial_rows):
            mv = max_val(trial_rows)
            if mv < best_max:
                best_max = mv
                best_rows = [list(r) for r in trial_rows]

    if best_rows is None:
        # Fallback: Pure greedy
        used = set()
        rows_greedy = []
        for _ in range(n_rows):
            row, used = greedy_row(used, mode=0)
            rows_greedy.append(row)
        return rows_greedy

    # ---------- Phase 2: Row-swap local search ----------
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for i in range(n_rows):
            if time.time() > deadline - 20:
                break
            other_rows = [r for j, r in enumerate(best_rows) if j != i]
            fixed_used = used_from_rows(other_rows)
            current_row = best_rows[i]
            new_row = find_replacement(fixed_used, current_row, best_max)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                if is_valid_global(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                        improved = True

    # ---------- Phase 3: Pair-swap refinement ----------
    if time.time() < deadline - 25:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        random.shuffle(pairs)
        for (i, j) in pairs:
            if time.time() > deadline - 15:
                break
            other_rows = [r for x, r in enumerate(best_rows) if x != i and x != j]
            fixed_used = used_from_rows(other_rows)
            
            # Try replacing the pair with different modes
            modes_pair = [(0, 0), (1, 1), (0, 1), (1, 0), (2, 2)]
            for m1, m2 in modes_pair:
                if time.time() > deadline - 15:
                    break
                used = set(fixed_used)
                r1, used = greedy_row(used, mode=m1)
                r2, used = greedy_row(used, mode=m2)
                
                new_rows = [r if x not in (i, j) else (r1 if x == i else r2)
                            for x, r in enumerate(best_rows)]
                if is_valid_global(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows

    # ---------- Phase 4: Simulated Annealing-style row replacement ----------
    current_rows = [r[:] for r in best_rows]
    current_row_diffs = [get_row_diffs(r) for r in current_rows]
    current_max = best_max
    t = 10.0
    cooling = 0.9995

    while time.time() < deadline - 5:
        if t < 0.05:
            t = 8.0  # Restart annealing

        idx = random.randrange(n_rows)

        # Calculate forbidden diffs
        forbidden = set()
        for i in range(n_rows):
            if i != idx:
                forbidden |= current_row_diffs[i]

        # Generate new row with random mode
        mode = random.choice([0, 1, 2])
        new_row, _ = greedy_row(forbidden, mode=mode)
        new_row_diffs = get_row_diffs(new_row)

        if new_row_diffs & forbidden:
            t *= cooling
            continue

        # Calculate new global max
        new_max = 0
        for i in range(n_rows):
            if i == idx:
                m = max(new_row)
            else:
                m = max(current_rows[i])
            if m > new_max:
                new_max = m

        # Decide acceptance
        if new_max < current_max:
            current_rows[idx] = new_row
            current_row_diffs[idx] = new_row_diffs
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        elif new_max == current_max:
            if random.random() < 0.2:
                current_rows[idx] = new_row
                current_row_diffs[idx] = new_row_diffs
        else:
            delta = new_max - current_max
            prob = pow(2.71828, -delta / t)
            if random.random() < prob:
                current_rows[idx] = new_row
                current_row_diffs[idx] = new_row_diffs
                current_max = new_max

        t *= cooling

    # ---------- Phase 5: Final polish with row-swap ----------
    if time.time() < deadline - 2:
        for i in range(n_rows):
            if time.time() > deadline - 1:
                break
            other_rows = [r for j, r in enumerate(best_rows) if j != i]
            fixed_used = used_from_rows(other_rows)
            new_row = find_replacement(fixed_used, best_rows[i], best_max)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                if is_valid_global(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows

    return best_rows
