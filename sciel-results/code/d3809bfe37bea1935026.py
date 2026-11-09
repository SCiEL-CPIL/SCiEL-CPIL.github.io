import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 118.0  # Safety margin from 120s limit (evaluator kills at 150s, but keep tight)

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2  # pairwise differences per row of length k+1
    
    # Bitmask helper: max diff is bounded by the scope. 
    # We will assume a max scope of ~200 for bitmask size, which is 1 << 200 (fits in Python int)
    # Actually, Python ints are arbitrary precision, so this is fine.
    
    def get_diff_mask(row):
        """Calculate the bitmask of all pairwise differences for a row."""
        mask = 0
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                mask |= (1 << d)
        return mask

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used_mask = 0
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            m = get_diff_mask(r)
            if m & used_mask:
                return False
            used_mask |= m
        return True

    # ---- Deterministic greedy (baseline, always valid) ----
    def greedy():
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Calculate diffs of x with existing row elements
                new_diffs_mask = 0
                for a in row:
                    new_diffs_mask |= (1 << (x - a))
                
                if not (new_diffs_mask & used_mask):
                    row.append(x)
                    used_mask |= new_diffs_mask
            rows.append(row)
        return rows

    # ---- Randomized Backtracking Constructor ----
    # Attempts to build a set of rows with max entry <= target_max
    def backtrack_construct(target_max, rng, n_rows, k):
        used_mask = 0
        rows = []
        
        # Order of rows: usually doesn't matter, but we can randomize row order if we track which row is which
        # Here we just build n_rows rows sequentially.
        
        def build_row(used_mask, target_max):
            """
            Builds a single row of length k+1 starting with 0, 
            avoiding diffs in used_mask, with max element <= target_max.
            Returns None if not possible.
            """
            row = [0]
            
            def _build(idx, current_row, current_row_mask):
                if idx == k:
                    return current_row[:]
                
                last = current_row[-1]
                # We need to choose the next element x > last
                # x must be <= target_max
                # The diffs x - a for a in current_row must not be in used_mask OR current_row_mask
                
                # Optimization: Iterate x from last+1 to target_max
                # To add randomness, we can shuffle the candidate list if it's not too large
                # But target_max can be ~100, so candidate list is ~100. Shuffling is cheap.
                
                candidates = []
                for x in range(last + 1, target_max + 1):
                    new_diffs_mask = 0
                    conflict = False
                    for a in current_row:
                        d = x - a
                        bit = 1 << d
                        # Check against global used_mask and local row mask
                        if (used_mask & bit) or (current_row_mask & bit):
                            conflict = True
                            break
                        new_diffs_mask |= bit
                    
                    if not conflict:
                        candidates.append((x, new_diffs_mask))
                
                if not candidates:
                    return None
                
                # Randomize order to escape local traps
                # If candidates are many, shuffle. If few, maybe keep sorted to prefer smaller?
                # For backtracking, randomization is key for diversity.
                rng.shuffle(candidates)
                
                for x, new_diffs_mask in candidates:
                    res = _build(idx + 1, current_row + [x], current_row_mask | new_diffs_mask)
                    if res is not None:
                        return res
                
                return None
            
            return _build(0, row, 0)

        for _ in range(n_rows):
            row = build_row(used_mask, target_max)
            if row is None:
                return None
            rows.append(row)
            used_mask |= get_diff_mask(row)
        
        return rows

    # ---- Optimized Simulated Annealing with Bitmasks ----
    def local_search(start, rng, steps, T0=5.0):
        cur = [list(r) for r in start]
        row_masks = [get_diff_mask(r) for r in cur]
        cur_used_mask = 0
        for m in row_masks:
            cur_used_mask |= m

        row_maxes = [max(r) for r in cur]
        cur_scope = max(row_maxes)
        # Objective: Minimize Scope. Tie-breaker: Sum of elements (to keep it tight)
        cur_sum = sum(sum(r) for r in cur)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0

        for it in range(steps):
            if it % 100 == 0 and remaining() < 5:
                break
            
            # Select row: bias towards rows with large max
            if rng.random() < 0.5:
                r = row_maxes.index(cur_scope)
            else:
                r = rng.randrange(n_rows)
            
            row = cur[r]
            other_used_mask = cur_used_mask & ~(row_masks[r])
            
            move_type = rng.random()
            
            new_row = None
            new_mask = None
            
            if move_type < 0.5:
                # MOVE 1: Targeted Compression of the Max Element
                p = len(row) - 1
                current_max = row[p]
                prev_val = row[p - 1]
                
                # Scan downwards
                for v in range(current_max - 1, prev_val, -1):
                    test_row = row[:]
                    test_row[p] = v
                    if test_row[p-1] >= v:
                        continue
                    nd_test = get_diff_mask(test_row)
                    # Check collision
                    if nd_test & other_used_mask:
                        continue
                    # Internal uniqueness is guaranteed if we just replace one element?
                    # No, replacing an element can create duplicate diffs within the row.
                    # e.g. row [0, 2, 4], replace 4 with 3 -> [0, 2, 3]. Diffs: 2,3,1. Unique.
                    # But if row [0, 1, 3], replace 3 with 2 -> [0, 1, 2]. Diffs: 1,2,1. Duplicate!
                    # So we must check len of set of diffs? 
                    # Bitmask doesn't tell us about duplicates directly unless we count bits?
                    # Actually, if we construct the mask, duplicates just set the same bit.
                    # We need to ensure all needed diffs are present and distinct.
                    # Since the row length is fixed, the number of pairs is fixed.
                    # If the mask has exactly 'needed' bits set, then all diffs are distinct.
                    if bin(nd_test).count("1") != needed:
                        continue
                        
                    new_row = test_row
                    new_mask = nd_test
                    break # Greedy: take first valid (smallest)
                
                if new_row is None:
                    continue

            elif move_type < 0.8:
                # MOVE 2: Random Interior Mutation
                p = rng.randrange(1, len(row))
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 10)
                
                if lo > hi:
                    continue
                
                span = hi - lo + 1
                v = lo + int(span * (rng.random() ** 1.5))
                
                if v == row[p]:
                    continue
                
                test_row = row[:]
                test_row[p] = v
                
                if (p > 0 and test_row[p-1] >= v) or (p < len(test_row)-1 and v >= test_row[p+1]):
                    continue
                    
                nd_test = get_diff_mask(test_row)
                if bin(nd_test).count("1") != needed:
                    continue
                if nd_test & other_used_mask:
                    continue
                
                new_row = test_row
                new_mask = nd_test

            else:
                # MOVE 3: Row Reconstruction (Structural)
                # Try to rebuild row avoiding other_used_mask, with tight scope
                # We use a simplified backtracking for a single row with a scope limit
                def rebuild_single_row(other_mask, limit):
                    row = [0]
                    def _bt(idx, cur_row, cur_mask):
                        if idx == k:
                            return cur_row[:]
                        last = cur_row[-1]
                        # Randomize candidates
                        cands = []
                        for x in range(last + 1, limit + 1):
                            nm = 0
                            ok = True
                            for a in cur_row:
                                d = x - a
                                bit = 1 << d
                                if (other_mask & bit) or (cur_mask & bit):
                                    ok = False
                                    break
                                nm |= bit
                            if ok:
                                cands.append((x, nm))
                        if not cands:
                            return None
                        rng.shuffle(cands)
                        for x, nm in cands:
                            res = _bt(idx+1, cur_row+[x], cur_mask|nm)
                            if res:
                                return res
                        return None
                    return _bt(0, row, 0)

                new_row = rebuild_single_row(other_used_mask, cur_scope)
                if new_row is None:
                    new_row = rebuild_single_row(other_used_mask, cur_scope + 5)
                if new_row is None:
                    continue
                new_mask = get_diff_mask(new_row)
                if bin(new_mask).count("1") != needed or (new_mask & other_used_mask):
                    continue

            if new_row is None:
                continue

            new_max_row = max(new_row)
            new_sum_row = sum(new_row)
            
            # Update global scope
            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)
            
            new_sum = cur_sum - sum(row) + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj
            
            if d <= 0 or (T > 0.01 and rng.random() < math.exp(-d / T)):
                # Accept
                cur[r] = new_row
                row_masks[r] = new_mask
                cur_used_mask = other_used_mask | new_mask
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]
            
            T = max(0.05, T * 0.9995)

        return best

    # ---- Main Search Loop ----
    # 1. Initial Greedy
    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    rng = random.Random(20240607)

    # 2. Diversify with Randomized Backtracking
    # Try to construct solutions with decreasing targets
    # Start from best_scope - 1 and go down
    # If found, update best and continue down
    
    # Also run SA in parallel/interleaved? 
    # Let's do a main loop:
    # 1. Try to find a better solution via Backtracking with target = best_scope - 1
    # 2. If found, update best.
    # 3. Run a short SA on the best solution to polish.
    # 4. Repeat.
    
    # Initial phase: Aggressive Backtracking
    # We try to find ANY solution with scope < best_scope
    # Then we tighten the target.
    
    # To be efficient, we can try multiple random restarts for a target.
    
    # Phase 1: Improve initial bound with Backtracking
    # Start target at best_scope - 1
    # If we find a solution, we can try to go lower.
    
    # However, backtracking might be slow if target is too tight.
    # Let's mix:
    # 50% time: Backtracking with target = best_scope - 1 (or lower if we found something)
    # 50% time: SA on best solution
    
    # Actually, let's just run a loop:
    while remaining() > 10:
        # Strategy: Try to beat current best_scope
        # Use Backtracking with target = best_scope - 1
        # If successful, update best_scope and continue trying to go lower.
        # If unsuccessful, run SA to try to find a better starting point or improve current best.
        
        # To avoid getting stuck in one mode, we alternate or use time slices.
        
        # Let's do:
        # 1. Try Backtracking with target = best_scope - 1 for a few seconds.
        # 2. If found, update best.
        # 3. Run SA on best for a few seconds.
        
        # Time slice 1: Backtracking
        t_start = time.time()
        found = False
        # Try a few restarts
        for _ in range(20):
            if time.time() - t_start > 2.0: # 2s budget for backtracking
                break
            # Target is best_scope - 1
            target = best_scope - 1
            # Add some randomness to target? No, we want to beat it.
            # Maybe try target = best_scope - 2 if we have time?
            # Let's stick to best_scope - 1 first.
            
            # Randomize the row order? 
            # The backtracker builds rows sequentially. 
            # We can shuffle the order in which we build rows? 
            # The `backtrack_construct` function builds rows 0 to n_rows-1.
            # We can pass a shuffled list of row indices? 
            # Actually, the rows are indistinguishable in the final output, 
            # but the construction order matters for the search space.
            # Let's just rely on the internal randomization of candidates.
            
            res = backtrack_construct(target, rng, n_rows, k)
            if res is not None:
                s = scope_of(res)
                if s < best_scope:
                    best_scope = s
                    best_rows = res
                    found = True
                    # If we found a solution, we can immediately try to go lower?
                    # Or let the next iteration handle it.
                    break
        
        # Time slice 2: SA Polish
        if remaining() > 5:
            # Run SA on the current best
            # Steps depend on time
            steps = 2000
            # If we just found a new best, maybe run longer SA?
            # Let's just run a fixed amount of time
            t_sa_start = time.time()
            # Estimate steps based on time? 
            # local_search doesn't have a time limit inside, only step count.
            # We need to adjust steps or add time check inside local_search.
            # The current local_search checks remaining() every 100 steps.
            # So we can just call it with a large step count and let it stop.
            res = local_search(best_rows, rng, 5000, T0=5.0)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    # Final safety check
    if not is_valid(best_rows):
        best_rows = base

    return best_rows
