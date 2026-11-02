import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a structural approach: row-replacement via random construction + simulated annealing,
    with a final greedy polish phase.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # ---- Helper functions ----
    
    def _row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def _greedy_init(n_rows, k, seed=42):
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

    def _compute_all_used(rows):
        used = set()
        for r in rows:
            used |= _row_diffs_set(r)
        return used

    def _global_max(rows):
        return max(max(r) for r in rows)

    def _max_except(rows, exclude_idx):
        m = 0
        for i, r in enumerate(rows):
            if i != exclude_idx:
                if r[-1] > m:
                    m = r[-1]
        return m

    def _random_valid_row(forbidden_diffs, max_limit, k, rng, max_candidates=5):
        """
        Construct a random valid row of length k+1 starting at 0, strictly increasing,
        with all internal differences distinct and disjoint from forbidden_diffs,
        with max entry <= max_limit.
        """
        row = [0]
        used_in_row = set()
        
        for _ in range(k):
            candidates = []
            start = row[-1] + 1
            end = max_limit + 1 if max_limit is not None else start + 300
            
            for x in range(start, end):
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden_diffs:
                    continue
                if new_diffs & used_in_row:
                    continue
                candidates.append(x)
                if len(candidates) >= max_candidates:
                    break
            
            if not candidates:
                return None
            
            chosen_x = rng.choice(candidates)
            row.append(chosen_x)
            used_in_row |= {chosen_x - a for a in row[:-1]}
        
        return row

    def _greedy_row_with_backtrack(forbidden_diffs, max_limit, k, max_depth=3):
        """
        Greedy construction of a row with limited backtracking.
        Returns a valid row or None.
        """
        def _try_fill(row, pos, used_local, depth):
            if pos == k + 1:
                return row[:]
            
            start = row[pos - 1] + 1
            end = max_limit + 1
            
            for v in range(start, end):
                new_diffs = {v - row[i] for i in range(pos)}
                if len(new_diffs) != pos:
                    continue
                if any(d in forbidden_diffs for d in new_diffs):
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
                    # Greedy fill for remaining positions
                    rest_row = row[:]
                    rest_used = used_local_new
                    for p in range(pos + 1, k + 1):
                        found = False
                        for v2 in range(rest_row[p-1] + 1, max_limit + 1):
                            nd2 = {v2 - rest_row[i] for i in range(p)}
                            if len(nd2) != p:
                                continue
                            if any(d in forbidden_diffs for d in nd2):
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

    def _reoptimize_row_greedy(row, other_diffs, k, max_scope):
        """
        Re-optimize a single row using greedy with backtracking against fixed differences.
        """
        if max_scope <= 0:
            return None
        
        result = _greedy_row_with_backtrack(set(other_diffs), max_scope, k, max_depth=3)
        return result

    # ---- Main algorithm ----
    
    # Initialize with greedy
    best = _greedy_init(n_rows, k, seed=42)
    best_val = _global_max(best)
    best_used = _compute_all_used(best)
    
    # Current state for SA
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    cur_used = set(best_used)

    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cool = 0.995
    
    no_improve_steps = 0
    max_no_improve = 50000
    rng = random.Random(12345)

    # SA loop: row-replacement based
    while time.time() < deadline and no_improve_steps < max_no_improve:
        # Pick a random row to replace
        ri = rng.randrange(n_rows)
        
        # Calculate forbidden diffs for this row
        forbidden = cur_used - cur_row_diffs[ri]
        
        # Try to find a new row with max <= cur_val (improvement or neutral)
        new_row = _random_valid_row(forbidden, cur_val, k, rng, max_candidates=8)
        
        if new_row is None:
            # Try with a slightly larger limit to allow moves that don't immediately improve
            new_row = _random_valid_row(forbidden, cur_val + 5, k, rng, max_candidates=8)
        
        if new_row is None:
            no_improve_steps += 1
            T = max(T_min, T * cool)
            continue
            
        # Efficient max calculation
        other_max = _max_except(cur, ri)
        new_max = max(other_max, new_row[-1])
        delta_e = new_max - cur_val

        accept = False
        if delta_e < 0:
            accept = True
        elif delta_e == 0:
            if rng.random() < 0.5:
                accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta_e / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            # Remove old diffs BEFORE updating cur_row_diffs
            old_diffs = cur_row_diffs[ri]
            cur_used -= old_diffs
            new_diffs = _row_diffs_set(new_row)
            cur_used |= new_diffs
            cur_row_diffs[ri] = new_diffs
            
            cur[ri] = new_row
            cur_val = new_max
            
            if cur_val < best_val:
                best_val = cur_val
                best = [r[:] for r in cur]
                best_used = set(cur_used)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1

        T = max(T_min, T * cool)

        # Periodic restart from best
        if no_improve_steps > 15000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            T = 5.0
            no_improve_steps = 0

    # Final polish: try to rebuild each row to minimize its max
    polish_deadline = deadline - 1.0
    for _ in range(5):
        if time.time() > polish_deadline:
            break
        for ri in range(n_rows):
            if time.time() > polish_deadline:
                break
            forbidden = best_used - _row_diffs_set(best[ri])
            # Try to find a row with max < best_val
            new_row = _random_valid_row(forbidden, best_val - 1, k, rng, max_candidates=10)
            if new_row is None:
                new_row = _greedy_row_with_backtrack(forbidden, best_val - 1, k, max_depth=3)
            if new_row is not None and new_row[-1] < best[ri][-1]:
                old_diffs = _row_diffs_set(best[ri])
                best_used -= old_diffs
                new_diffs = _row_diffs_set(new_row)
                best_used |= new_diffs
                best[ri] = new_row
                best_val = _global_max(best)

    # One more pass of SA with the polished best
    if time.time() < deadline - 2.0:
        cur = [r[:] for r in best]
        cur_val = best_val
        cur_row_diffs = [_row_diffs_set(r) for r in cur]
        cur_used = set(best_used)
        
        T = 3.0
        no_improve_steps = 0
        sa_deadline = deadline - 0.5
        
        while time.time() < sa_deadline and no_improve_steps < 30000:
            ri = rng.randrange(n_rows)
            forbidden = cur_used - cur_row_diffs[ri]
            
            new_row = _random_valid_row(forbidden, cur_val, k, rng, max_candidates=8)
            if new_row is None:
                new_row = _random_valid_row(forbidden, cur_val + 3, k, rng, max_candidates=8)
            
            if new_row is None:
                no_improve_steps += 1
                T = max(T_min, T * cool)
                continue
            
            other_max = _max_except(cur, ri)
            new_max = max(other_max, new_row[-1])
            delta_e = new_max - cur_val

            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                if rng.random() < 0.5:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta_e / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                old_diffs = cur_row_diffs[ri]
                cur_used -= old_diffs
                new_diffs = _row_diffs_set(new_row)
                cur_used |= new_diffs
                cur_row_diffs[ri] = new_diffs
                
                cur[ri] = new_row
                cur_val = new_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    best_used = set(cur_used)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1

            T = max(T_min, T * cool)

    return best

# EVOLVE-BLOCK-END
