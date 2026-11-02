import time
import random
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: multi-start greedy initialization, then randomized local search
    with simulated annealing, row re-solve moves, and a final high-precision polish pass.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # ---- Helper functions ----
    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_used(rows):
        all_used = set()
        for row in rows:
            all_used |= _row_diffs(row)
        return all_used

    def _global_max(rows):
        return max(max(row) for row in rows)

    def _is_sidon(row):
        """Check if all pairwise differences in the row are distinct."""
        n = len(row)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
        return True

    def greedy_construct(n_rows, k, rng=None):
        """Deterministic or randomized greedy construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                if rng is not None:
                    # Randomized: pick from a window of valid candidates
                    valid_xs = []
                    scan = x + 1
                    while scan <= min(x + 30, 500) and len(valid_xs) < 5:
                        new_diffs = {scan - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & used):
                            valid_xs.append(scan)
                        scan += 1
                    if not valid_xs:
                        # Fallback: scan further
                        scan = x + 1
                        while scan <= 5000:
                            new_diffs = {scan - a for a in row}
                            if len(new_diffs) == len(row) and not (new_diffs & used):
                                valid_xs.append(scan)
                                break
                            scan += 1
                        if not valid_xs:
                            break
                    # Weighted pick biased toward smaller
                    weights = [1.0 / (i + 1) for i in range(len(valid_xs))]
                    total = sum(weights)
                    r = rng.random() * total
                    cum = 0.0
                    chosen = valid_xs[-1]
                    for v, w in zip(valid_xs, weights):
                        cum += w
                        if r <= cum:
                            chosen = v
                            break
                    row.append(chosen)
                    for a in row[:-1]:
                        used.add(chosen - a)
                    x = chosen
                else:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    def _resolve_row_given_others(ri, rows, k, max_target, rng=None):
        """Rebuild row ri greedily, avoiding only other rows' diffs.
        Returns new row (max <= max_target) or None."""
        other_used = set()
        for i, r in enumerate(rows):
            if i == ri:
                continue
            other_used |= _row_diffs(r)
        
        fresh_used = set(other_used)
        row = [0]
        x = 0
        limit = max_target
        while len(row) < k + 1:
            if rng is not None:
                valid_xs = []
                scan = x + 1
                while scan <= min(x + 25, limit) and len(valid_xs) < 4:
                    new_diffs = {scan - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & fresh_used):
                        valid_xs.append(scan)
                    scan += 1
                if not valid_xs:
                    scan = x + 1
                    while scan <= limit:
                        new_diffs = {scan - a for a in row}
                        if len(new_diffs) == len(row) and not (new_diffs & fresh_used):
                            valid_xs.append(scan)
                            break
                        scan += 1
                    if not valid_xs:
                        return None
                weights = [1.0 / (i + 1) for i in range(len(valid_xs))]
                total = sum(weights)
                r = rng.random() * total
                cum = 0.0
                chosen = valid_xs[-1]
                for v, w in zip(valid_xs, weights):
                    cum += w
                    if r <= cum:
                        chosen = v
                        break
                row.append(chosen)
                for a in row[:-1]:
                    fresh_used.add(chosen - a)
                x = chosen
            else:
                x += 1
                if x > limit:
                    return None
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & fresh_used):
                    row.append(x)
                    fresh_used |= new_diffs
        return row

    def simulated_annealing(rows, n_rows, k, start_time, deadline, rng):
        """SA-based local search with row re-solve moves to minimize the maximum entry."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 400.0
        T_min = 0.3
        cool = 0.99997
        
        no_improve = 0
        max_no_improve = 20000
        restarts = 0
        max_restarts = 25
        
        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Choose move type
            move_type = rng.random()
            if move_type < 0.55:
                # Single position move
                max_r = _max_rows()
                if max_r and rng.random() < 0.6:
                    ri = rng.choice(max_r)
                else:
                    ri = rng.randrange(n_rows)
                pi = rng.randrange(1, k + 1)
                
                old_val = current_rows[ri][pi]
                lo = current_rows[ri][pi - 1] + 1
                hi = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
                
                candidates = []
                # Decreasing candidates
                max_decrease = min(50, old_val - lo)
                for d in range(1, max_decrease + 1):
                    candidates.append(old_val - d)
                # Increasing candidates
                max_increase = min(15, hi - old_val)
                for d in range(1, max_increase + 1):
                    candidates.append(old_val + d)
                # Random jump
                if rng.random() < 0.2 and lo <= hi:
                    candidates.append(rng.randint(lo, hi))
                
                if not candidates:
                    no_improve += 1
                    T = max(T_min, T * cool)
                    continue
                
                candidates.sort(key=lambda v: (v > old_val, v))
                
                moved = False
                for new_val in candidates:
                    if new_val <= current_rows[ri][pi - 1]:
                        continue
                    if pi < k and new_val >= current_rows[ri][pi + 1]:
                        continue
                    
                    cand_row = current_rows[ri][:]
                    cand_row[pi] = new_val
                    
                    # Validate
                    new_diffs = set()
                    valid = True
                    n = len(cand_row)
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = cand_row[j] - cand_row[i]
                            if d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    # Check cross-row conflicts
                    for d in new_diffs:
                        if d in all_used and d not in row_diffs[ri]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    cand_row_max = max(cand_row)
                    old_row_max = row_maxes[ri]
                    row_maxes[ri] = cand_row_max
                    new_global_max = max(row_maxes)
                    row_maxes[ri] = old_row_max
                    
                    delta = new_global_max - current_val
                    
                    accept = False
                    if delta < 0:
                        accept = True
                    elif delta == 0:
                        if rng.random() < 0.25:
                            accept = True
                    else:
                        if T > T_min:
                            prob = math.exp(-delta / max(T, 0.1))
                            if rng.random() < prob:
                                accept = True
                    
                    if accept:
                        all_used -= row_diffs[ri]
                        all_used |= new_diffs
                        current_rows[ri] = cand_row
                        row_diffs[ri] = new_diffs
                        row_maxes[ri] = cand_row_max
                        current_val = max(row_maxes)
                        
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        moved = True
                        break
                
                if not moved:
                    no_improve += 1
            
            elif move_type < 0.85:
                # Row re-solve move: rebuild one row from scratch
                max_r = _max_rows()
                if max_r and rng.random() < 0.7:
                    ri = rng.choice(max_r)
                else:
                    ri = rng.randrange(n_rows)
                
                old_row_max = row_maxes[ri]
                target = old_row_max + rng.randint(-5, 5)
                new_row = _resolve_row_given_others(ri, current_rows, k, target, rng)
                
                if new_row is not None:
                    new_diffs = _row_diffs(new_row)
                    # Check no conflicts with other rows
                    other_used = set()
                    for rj in range(n_rows):
                        if rj != ri:
                            other_used |= row_diffs[rj]
                    
                    if not (new_diffs & other_used):
                        new_row_max = max(new_row)
                        row_maxes[ri] = new_row_max
                        new_global_max = max(row_maxes)
                        row_maxes[ri] = old_row_max
                        
                        delta = new_global_max - current_val
                        
                        accept = False
                        if delta < 0:
                            accept = True
                        elif delta == 0:
                            if rng.random() < 0.3:
                                accept = True
                        else:
                            if T > T_min:
                                prob = math.exp(-delta / max(T, 0.1))
                                if rng.random() < prob:
                                    accept = True
                        
                        if accept:
                            all_used -= row_diffs[ri]
                            all_used |= new_diffs
                            current_rows[ri] = new_row
                            row_diffs[ri] = new_diffs
                            row_maxes[ri] = new_row_max
                            current_val = max(row_maxes)
                            
                            if current_val < best_val:
                                best_val = current_val
                                best_rows = [r[:] for r in current_rows]
                                no_improve = 0
                            else:
                                no_improve += 1
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            else:
                # Two-position move in same row
                max_r = _max_rows()
                if max_r and rng.random() < 0.5:
                    ri = rng.choice(max_r)
                else:
                    ri = rng.randrange(n_rows)
                
                positions = list(range(1, k + 1))
                rng.shuffle(positions)
                pi1 = positions[0]
                pi2 = positions[1] if len(positions) > 1 else None
                
                if pi2 is None:
                    no_improve += 1
                    T = max(T_min, T * cool)
                    continue
                
                # Try moving both positions
                old_v1 = current_rows[ri][pi1]
                old_v2 = current_rows[ri][pi2]
                
                # Generate small random offsets
                dv1 = rng.randint(-10, 10)
                dv2 = rng.randint(-10, 10)
                new_v1 = old_v1 + dv1
                new_v2 = old_v2 + dv2
                
                lo1 = current_rows[ri][pi1 - 1] + 1
                hi1 = (current_rows[ri][pi1 + 1] - 1) if pi1 < k else current_val + 20
                lo2 = current_rows[ri][pi2 - 1] + 1
                hi2 = (current_rows[ri][pi2 + 1] - 1) if pi2 < k else current_val + 20
                
                if not (lo1 <= new_v1 <= hi1 and lo2 <= new_v2 <= hi2):
                    no_improve += 1
                    T = max(T_min, T * cool)
                    continue
                
                # Ensure ordering is maintained
                if pi1 < pi2:
                    if new_v1 >= new_v2:
                        no_improve += 1
                        T = max(T_min, T * cool)
                        continue
                else:
                    if new_v2 >= new_v1:
                        no_improve += 1
                        T = max(T_min, T * cool)
                        continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi1] = new_v1
                cand_row[pi2] = new_v2
                
                # Check full ordering
                valid = True
                for idx in range(len(cand_row) - 1):
                    if cand_row[idx] >= cand_row[idx + 1]:
                        valid = False
                        break
                
                if not valid:
                    no_improve += 1
                    T = max(T_min, T * cool)
                    continue
                
                new_diffs = set()
                n = len(cand_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = cand_row[j] - cand_row[i]
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                
                if not valid:
                    no_improve += 1
                    T = max(T_min, T * cool)
                    continue
                
                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        valid = False
                        break
                
                if not valid:
                    no_improve += 1
                    T = max(T_min, T * cool)
                    continue
                
                cand_row_max = max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if rng.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = max(row_maxes)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            T = max(T_min, T * cool)
            
            # Restart if stuck
            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                all_used = _all_used(current_rows)
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = 250.0
                no_improve = 0
        
        return best_rows, best_val

    def row_reopt_polish(rows, n_rows, k, deadline, rng):
        """
        Polish by re-optimizing each row against the others with tight scope.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(n_rows):
                if time.time() > deadline:
                    break
                all_used_excl = set()
                for rj in range(n_rows):
                    if rj != ri:
                        all_used_excl |= row_diffs[rj]
                
                # Try to re-optimize this row with tighter scope
                new_row = _resolve_row_given_others(ri, current_rows, k, current_val - 1, rng)
                
                if new_row is not None:
                    new_row_max = max(new_row)
                    old_row_max = max(current_rows[ri])
                    
                    if new_row_max < old_row_max:
                        new_diffs = _row_diffs(new_row)
                        all_used -= row_diffs[ri]
                        all_used |= new_diffs
                        current_rows[ri] = new_row
                        row_diffs[ri] = new_diffs
                        current_val = _global_max(current_rows)
                        improved = True
        
        return current_rows, current_val

    # Run multiple SA trials with different seeds
    for trial in range(15):
        if time.time() > deadline - 10:
            break
        
        seed = trial * 7919 + 12345
        rng = random.Random(seed)
        
        # Multi-start: use both deterministic and randomized greedy
        if trial % 3 == 0:
            init_rows = greedy_construct(n_rows, k, rng=None)
        else:
            init_rows = greedy_construct(n_rows, k, rng=rng)
        
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        # Give each trial a fair time budget
        trial_deadline = min(deadline - 10, start_time + 145.0)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline, rng)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows

    # Final polish with remaining time
    if time.time() < deadline - 3:
        polish_deadline = min(deadline, start_time + 145.0)
        rng = random.Random(99999)
        
        # First, do SA polish
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline, rng
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows
        
        # Then, do row re-optimization polish
        if time.time() < deadline - 1:
            reopt_rows, reopt_scope = row_reopt_polish(
                [r[:] for r in best_solution], n_rows, k, deadline - 1, rng
            )
            if reopt_scope < best_scope:
                best_scope = reopt_scope
                best_solution = reopt_rows
        
        # Final tightening: try to reduce scope by 1 repeatedly
        if time.time() < deadline - 0.5:
            current = [r[:] for r in best_solution]
            cur_scope = _global_max(current)
            for _ in range(5):
                if time.time() > deadline - 0.5:
                    break
                progress = False
                all_used = _all_used(current)
                row_diffs = [_row_diffs(r) for r in current]
                for ri in range(n_rows):
                    if time.time() > deadline - 0.5:
                        break
                    all_used_excl = set()
                    for rj in range(n_rows):
                        if rj != ri:
                            all_used_excl |= row_diffs[rj]
                    target = max(current[ri]) - 1
                    if target <= 0:
                        continue
                    new_row = _resolve_row_given_others(ri, current, k, target, rng)
                    if new_row is not None and max(new_row) <= target:
                        new_diffs = _row_diffs(new_row)
                        all_used -= row_diffs[ri]
                        all_used |= new_diffs
                        current[ri] = new_row
                        row_diffs[ri] = new_diffs
                        progress = True
                if not progress:
                    break
            final_scope = _global_max(current)
            if final_scope < best_scope:
                best_scope = final_scope
                best_solution = current

    return best_solution
# EVOLVE-BLOCK-END
