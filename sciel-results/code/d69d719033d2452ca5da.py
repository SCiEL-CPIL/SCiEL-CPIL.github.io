import time
import random
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a hybrid approach: greedy initialization, then randomized local search
    with simulated annealing and row-level re-optimization to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # ---- Greedy fallback (always valid) ----
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

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

    def local_search_init(n_rows, k, seed=None):
        """Greedy initialization with randomization."""
        if seed is not None:
            random.seed(seed)
        
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def _reoptimize_row(row, other_diffs, k, max_scope=None):
        """
        Globally re-optimize a single row against fixed differences from other rows.
        Uses iterative local search on the row elements to minimize the max entry
        while avoiding conflicts with other_diffs and internal conflicts.
        Returns the improved row.
        """
        n = k + 1
        best_row = list(row)
        best_max = max(best_row)
        
        if max_scope is not None and best_max > max_scope:
            return best_row
        
        other_set = set(other_diffs)
        
        for iteration in range(100):
            improved = False
            for pi in range(1, n):
                old_val = best_row[pi]
                min_allowed = best_row[pi - 1] + 1
                max_allowed = (best_row[pi + 1] - 1) if pi < n - 1 else (max_scope if max_scope else old_val)
                
                # Try decreasing values first
                for new_val in range(min(old_val - 1, max_allowed), min_allowed - 1, -1):
                    if new_val < min_allowed:
                        break
                    if pi < n - 1 and new_val >= best_row[pi + 1]:
                        break
                    cand = best_row[:]
                    cand[pi] = new_val
                    valid = True
                    diffs = set()
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = cand[j] - cand[i]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                    if not valid:
                        continue
                    if diffs & other_set:
                        continue
                    new_max = max(cand)
                    if new_max < best_max:
                        best_row = cand
                        best_max = new_max
                        improved = True
                        break
                    elif new_max == best_max:
                        if random.random() < 0.2:
                            best_row = cand
                            improved = True
                            break
            if not improved:
                break
        return best_row

    def _row_reopt_bt(row_init, all_used_excl, k, max_scope, max_depth=3):
        """
        Re-optimize a single row using greedy with limited backtracking.
        """
        def _try_fill(row, pos, used_local, depth):
            if pos == k + 1:
                return row[:]
            
            start = row[pos - 1] + 1
            end = max_scope + 1
            
            for v in range(start, end):
                new_diffs = {v - row[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue
                
                row[pos] = v
                used_local_new = used_local | new_diffs
                
                if depth < max_depth:
                    result = _try_fill(row, pos + 1, used_local_new, depth + 1)
                    if result is not None:
                        return result
                else:
                    rest_row = row[:]
                    rest_used = used_local_new
                    for p in range(pos + 1, k + 1):
                        found = False
                        for v2 in range(rest_row[p-1] + 1, max_scope + 1):
                            nd2 = {v2 - rest_row[i] for i in range(p)}
                            if len(nd2) != p:
                                continue
                            if any(d in all_used_excl for d in nd2):
                                continue
                            if any(d in rest_used for d in nd2):
                                continue
                            rest_row.append(v2)
                            rest_used |= nd2
                            found = True
                            break
                        if not found:
                            return None
                    return rest_row
            
            return None
        
        row = [0] + [0] * k
        result = _try_fill(row, 1, set(), 0)
        return result

    def _row_reopt(row, all_used_excl, k, max_scope=None):
        """
        Re-optimize a single row given the differences used by all other rows.
        """
        if max_scope is None:
            max_scope = 200
        
        row_greedy = [0]
        used_local = set()
        for pos in range(1, k + 1):
            found = False
            start = row_greedy[pos - 1] + 1
            end = max_scope + 1
            
            for v in range(start, end):
                new_diffs = {v - row_greedy[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in all_used_excl for d in new_diffs):
                    continue
                if any(d in used_local for d in new_diffs):
                    continue
                
                row_greedy.append(v)
                used_local |= new_diffs
                found = True
                break
            
            if not found:
                return None
        
        bt_result = _row_reopt_bt(row_greedy, all_used_excl, k, max_scope, max_depth=2)
        if bt_result is not None:
            return bt_result
        
        return row_greedy

    def row_reopt_polish(rows, n_rows, k, deadline):
        """
        Polish by re-optimizing each row against the others.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(n_rows):
                all_used_excl = set()
                for rj in range(n_rows):
                    if rj != ri:
                        all_used_excl |= row_diffs[rj]
                
                new_row = _row_reopt(current_rows[ri], all_used_excl, k, max_scope=current_val - 1)
                
                if new_row is not None:
                    new_row_max = max(new_row)
                    old_row_max = max(current_rows[ri])
                    
                    if new_row_max < old_row_max:
                        all_used -= row_diffs[ri]
                        new_diffs = _row_diffs(new_row)
                        all_used |= new_diffs
                        current_rows[ri] = new_row
                        row_diffs[ri] = new_diffs
                        current_val = _global_max(current_rows)
                        improved = True
                        all_used_excl2 = set()
                        for rj in range(n_rows):
                            if rj != ri:
                                all_used_excl2 |= row_diffs[rj]
                        new_row2 = _row_reopt(current_rows[ri], all_used_excl2, k, max_scope=current_val - 1)
                        if new_row2 is not None and max(new_row2) < max(current_rows[ri]):
                            all_used -= row_diffs[ri]
                            new_diffs2 = _row_diffs(new_row2)
                            all_used |= new_diffs2
                            current_rows[ri] = new_row2
                            row_diffs[ri] = new_diffs2
                            current_val = _global_max(current_rows)
        
        return current_rows, current_val

    def simulated_annealing(rows, n_rows, k, start_time, deadline):
        """SA-based local search with row-level re-optimization."""
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        row_maxes = [max(r) for r in current_rows]
        all_used = _all_used(current_rows)
        row_diffs = [_row_diffs(r) for r in current_rows]
        
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        T = 500.0
        T_min = 0.5
        cool = 0.99998
        
        no_improve = 0
        max_no_improve = 30000
        restarts = 0
        max_restarts = 25
        last_reopt_time = time.time()
        reopt_interval = 1.2
        
        def _max_rows():
            return [i for i in range(n_rows) if row_maxes[i] == current_val]
        
        def _other_diffs_except(ri):
            other = set()
            for j in range(n_rows):
                if j != ri:
                    other |= row_diffs[j]
            return other
        
        while time.time() < deadline and no_improve < max_no_improve:
            now = time.time()
            if now - last_reopt_time > reopt_interval and no_improve > 200:
                last_reopt_time = now
                target_row = max(range(n_rows), key=lambda i: row_maxes[i])
                other_diffs = _other_diffs_except(target_row)
                improved_row = _reoptimize_row(current_rows[target_row], other_diffs, k, max_scope=best_val - 1)
                if max(improved_row) < row_maxes[target_row]:
                    new_diffs = _row_diffs(improved_row)
                    all_used -= row_diffs[target_row]
                    all_used |= new_diffs
                    current_rows[target_row] = improved_row
                    row_diffs[target_row] = new_diffs
                    row_maxes[target_row] = max(improved_row)
                    current_val = max(row_maxes)
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    continue
            
            max_r = _max_rows()
            if max_r and random.random() < 0.6:
                ri = random.choice(max_r)
            else:
                ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            candidates = []
            
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(80, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 50
            max_increase = min(30, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
            
            if random.random() < 0.25:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            if random.random() < 0.15 and k >= 2:
                pj = random.randrange(1, k + 1)
                if pj != pi:
                    if pj > pi:
                        if current_rows[ri][pi] < current_rows[ri][pj] and \
                           (pi == 1 or current_rows[ri][pi-1] < current_rows[ri][pj]) and \
                           (pj == k or current_rows[ri][pj-1] < current_rows[ri][pi]):
                            candidates.append(-current_rows[ri][pj])
                    else:
                        if current_rows[ri][pj] < current_rows[ri][pi] and \
                           (pj == 1 or current_rows[ri][pj-1] < current_rows[ri][pi]) and \
                           (pi == k or current_rows[ri][pi-1] < current_rows[ri][pj]):
                            candidates.append(-current_rows[ri][pi])
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            candidates.sort(key=lambda v: (v > old_val, v))
            
            moved = False
            for new_val in candidates:
                if new_val < 0:
                    swap_target = -new_val
                    pj = None
                    for idx in range(1, k+1):
                        if current_rows[ri][idx] == swap_target:
                            pj = idx
                            break
                    if pj is None or pj == pi:
                        continue
                    
                    cand_row = current_rows[ri][:]
                    cand_row[pi], cand_row[pj] = cand_row[pj], cand_row[pi]
                    
                    valid = True
                    for idx in range(len(cand_row) - 1):
                        if cand_row[idx] >= cand_row[idx+1]:
                            valid = False
                            break
                    
                    if not valid:
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
                        continue
                    
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
                        if random.random() < 0.3:
                            accept = True
                    else:
                        if T > T_min:
                            prob = math.exp(-delta / max(T, 0.1))
                            if random.random() < prob:
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
                    continue
                
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
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
                    if random.random() < 0.2:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
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
            
            T = max(T_min, T * cool)
            
            if no_improve > max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                current_rows = [r[:] for r in best_rows]
                current_val = best_val
                all_used = _all_used(current_rows)
                row_diffs = [_row_diffs(r) for r in current_rows]
                row_maxes = [max(r) for r in current_rows]
                T = 300.0
                no_improve = 0
        
        return best_rows, best_val

    # Run multiple SA trials with different seeds
    for trial in range(12):
        if time.time() > deadline - 8:
            break
        
        seed = trial * 7919 + 12345
        init_rows = local_search_init(n_rows, k, seed=seed)
        init_scope = _global_max(init_rows)
        
        if init_scope < best_scope:
            best_scope = init_scope
            best_solution = [r[:] for r in init_rows]
        
        trial_deadline = min(deadline - 8, start_time + 145.0)
        sa_rows, sa_scope = simulated_annealing(init_rows, n_rows, k, start_time, trial_deadline)
        
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_rows

    # Final polish with remaining time
    if time.time() < deadline - 3:
        polish_deadline = min(deadline, start_time + 145.0)
        
        polish_rows, polish_scope = simulated_annealing(
            [r[:] for r in best_solution], n_rows, k, start_time, polish_deadline
        )
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polish_rows
        
        if time.time() < deadline - 1:
            reopt_rows, reopt_scope = row_reopt_polish(
                [r[:] for r in best_solution], n_rows, k, deadline - 1
            )
            if reopt_scope < best_scope:
                best_scope = reopt_scope
                best_solution = reopt_rows
        
        if time.time() < deadline - 0.5:
            current = [r[:] for r in best_solution]
            cur_scope = _global_max(current)
            for _ in range(8):
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
                    new_row = _row_reopt(current[ri], all_used_excl, k, max_scope=target)
                    if new_row is not None and max(new_row) <= target:
                        all_used -= row_diffs[ri]
                        new_diffs = _row_diffs(new_row)
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
