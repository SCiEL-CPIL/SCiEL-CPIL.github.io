"""
Difference Triangle Set Solver (7, 6)
Uses Simulated Annealing with a CSP-based polish phase for the worst row.
"""
import time
import random
import math

# EVOLVE-BLOCK-START

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

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

def _greedy_row_with_update(used, k, max_val=None, rng=None):
    """Build a single row greedily (optionally randomized), updating used as we go."""
    row = [0]
    x = 0
    limit = max_val if max_val is not None else 10000
    while len(row) < k + 1:
        if rng is None:
            x += 1
            if x > limit:
                return None, used
            new_diffs = [x - a for a in row]
            if any(d in used for d in new_diffs):
                continue
            if len(new_diffs) != len(set(new_diffs)):
                continue
            row.append(x)
            for d in new_diffs:
                used.add(d)
        else:
            valid_xs = []
            scan = x + 1
            while scan <= min(x + 40, limit) and len(valid_xs) < 6:
                new_diffs = [scan - a for a in row]
                if not any(d in used for d in new_diffs) and len(new_diffs) == len(set(new_diffs)):
                    valid_xs.append(scan)
                scan += 1
            if not valid_xs:
                scan = x + 1
                while scan <= limit:
                    new_diffs = [scan - a for a in row]
                    if not any(d in used for d in new_diffs) and len(new_diffs) == len(set(new_diffs)):
                        valid_xs.append(scan)
                        break
                    scan += 1
                if not valid_xs:
                    return None, used
            
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
    return row, used

def _build_greedy(n_rows, k, rng=None):
    """Fast greedy: each row takes smallest (or random-window) next mark with no diff collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = _greedy_row_with_update(used, k, rng=rng)
        if row is None:
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = [x - a for a in row]
                if not any(d in used for d in new) and len(new) == len(set(new)):
                    row.append(x)
                    used |= set(new)
        rows.append(row)
    return rows

def _csp_solve_row(forbidden, k, target_max, rng):
    """
    Use backtracking to find a valid row of length k+1 starting at 0,
    with max <= target_max, avoiding differences in 'forbidden'.
    """
    row = [0]
    used_diffs = set()
    
    # Pre-compute forbidden as a set for O(1) lookup
    forbidden_set = set(forbidden)
    
    def backtrack(idx, last_val):
        if idx == k + 1:
            return row[:]
        
        start = last_val + 1
        end = target_max
        
        # Generate candidates
        # To make it faster, we can limit the number of candidates or use randomization
        # For small k, full scan is okay, but we can shuffle
        candidates = list(range(start, end + 1))
        if len(candidates) > 20:
            # Randomize to avoid getting stuck in same local minima
            rng.shuffle(candidates)
        else:
            rng.shuffle(candidates)
            
        for v in candidates:
            # Check diffs with previous elements
            new_diffs = []
            ok = True
            for i in range(idx):
                d = v - row[i]
                if d in forbidden_set or d in used_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            
            if ok:
                row.append(v)
                used_diffs.update(new_diffs)
                
                res = backtrack(idx + 1, v)
                if res is not None:
                    return res
                
                row.pop()
                used_diffs.difference_update(new_diffs)
        
        return None
    
    return backtrack(1, 0)

def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    T = 10.0
    decay = 0.999
    min_T = 0.01

    t0 = time.time()
    steps = 0
    
    while time.time() - t0 < time_budget:
        steps += 1
        r = rng.random()
        
        if r < 0.5:
            # Single position move
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            row = cur[ri]
            old_val = row[pi]
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else 10**6
            if hi < lo:
                T = max(min_T, T * decay)
                continue
            
            # Generate candidates
            candidates = []
            for d in range(-5, 6):
                nv = old_val + d
                if lo <= nv <= hi and nv != old_val:
                    candidates.append(nv)
            if hi - lo > 20:
                for _ in range(5):
                    candidates.append(rng.randint(lo, hi))
            
            if not candidates:
                T = max(min_T, T * decay)
                continue
                
            rng.shuffle(candidates)
            moved = False
            for nv in candidates:
                cand_row = row[:]
                cand_row[pi] = nv
                if not _is_sidon(cand_row):
                    continue
                
                new_diffs = _row_diffs_set(cand_row)
                old_diffs = _row_diffs_set(row)
                other_diffs = cur_used - old_diffs
                
                if new_diffs & other_diffs:
                    continue
                
                # Valid move
                old_max = cur_val
                new_max = max(nv, max(row[i] for i in range(k+1) if i != pi), max(max(r) for i, r in enumerate(cur) if i != ri))
                delta = new_max - old_max
                
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = cand_row
                    cur_val = new_max
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
                    moved = True
                    break
            
        elif r < 0.9:
            # Re-resolve row with CSP
            ri = rng.randrange(n_rows)
            old_row = cur[ri]
            old_max = max(old_row)
            # Try to improve max by 1-3
            target = old_max - rng.randint(1, 3)
            if target < k: # Minimum possible max for k=6 is 6 (0,1,2,3,4,5,6)
                target = old_max
                
            other_used = set()
            for i, r in enumerate(cur):
                if i != ri:
                    other_used |= _row_diffs_set(r)
            
            new_row = _csp_solve_row(other_used, k, target, rng)
            
            if new_row is not None:
                new_diffs = _row_diffs_set(new_row)
                if not (new_diffs & other_used):
                    old_diffs = _row_diffs_set(old_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    new_max = max(max(r) for r in cur)
                    delta = new_max - cur_val
                    if delta <= 0 or rng.random() < math.exp(-delta / T):
                        cur_val = new_max
                        if cur_val < best_val:
                            best_val = cur_val
                            best = [r2[:] for r2 in cur]
        else:
            # Kick: Random restart from best with perturbation
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            for _ in range(rng.randint(1, 2)):
                ri = rng.randrange(n_rows)
                pi = rng.randrange(1, k + 1)
                row = cur[ri]
                lo = row[pi - 1] + 1
                hi = row[pi + 1] - 1 if pi < k else 10**6
                if hi >= lo:
                    nv = rng.randint(lo, hi)
                    cand_row = row[:]
                    cand_row[pi] = nv
                    if _is_sidon(cand_row):
                        new_diffs = _row_diffs_set(cand_row)
                        old_diffs = _row_diffs_set(row)
                        other_diffs = cur_used - old_diffs
                        if not (new_diffs & other_diffs):
                            cur_used -= old_diffs
                            cur_used |= new_diffs
                            cur[ri] = cand_row
                            cur_val = _max_entry(cur)

        T = max(min_T, T * decay)
        
        # Re-anchoring
        if steps % 1000 == 0 and cur_val > best_val + 10:
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val

    return best, best_val

def _polish_solution(rows, n_rows, k, time_budget, seed):
    """
    Polish phase: Iteratively try to reduce the max of the worst row using CSP.
    """
    rng = random.Random(seed)
    cur = [r[:] for r in rows]
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    
    t0 = time.time()
    
    while time.time() - t0 < time_budget:
        # Find worst row
        max_val = -1
        worst_idx = -1
        for i, r in enumerate(cur):
            if max(r) > max_val:
                max_val = max(r)
                worst_idx = i
        
        # Try to reduce max_val
        target = max_val - 1
        if target < k:
            break
            
        other_used = set()
        for i, r in enumerate(cur):
            if i != worst_idx:
                other_used |= _row_diffs_set(r)
        
        new_row = _csp_solve_row(other_used, k, target, rng)
        
        if new_row is not None:
            new_diffs = _row_diffs_set(new_row)
            if not (new_diffs & other_used):
                old_diffs = _row_diffs_set(cur[worst_idx])
                # Update cur
                # We don't maintain global used set here for speed, just check validity
                cur[worst_idx] = new_row
                new_max = _max_entry(cur)
                if new_max < best_val:
                    best_val = new_max
                    best = [r2[:] for r2 in cur]
                # Continue polishing with new worst row
                continue
        
        # If we can't reduce max, try to find a different configuration with same max
        # that might allow reduction later? Or just break to save time.
        # Let's try a few random single-position moves on the worst row
        moved = False
        for _ in range(10):
            ri = worst_idx
            pi = rng.randrange(1, k + 1)
            row = cur[ri]
            old_val = row[pi]
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else 10**6
            if hi < lo: continue
            nv = rng.randint(lo, hi)
            if nv == old_val: continue
            cand_row = row[:]
            cand_row[pi] = nv
            if _is_sidon(cand_row):
                new_diffs = _row_diffs_set(cand_row)
                old_diffs = _row_diffs_set(row)
                other_diffs = other_used - old_diffs
                if not (new_diffs & other_diffs):
                    cur[ri] = cand_row
                    moved = True
                    break
        
        if not moved:
            # Stuck on this worst row. Try next worst?
            # Or break to avoid infinite loop.
            # Let's try to reduce the second worst row?
            # For simplicity, if we can't improve the worst, we might be at a local optimum.
            # Break to return best found.
            break
            
    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Phase 1: Diverse initializations
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(5):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(1000 + s)))
    
    inits.sort(key=_max_entry)
    top_inits = inits[:3]

    # Phase 2: SA
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    n_sa = 3
    budget_per_sa = max(5.0, remaining / n_sa - 2.0)

    overall_best = None
    overall_best_val = float('inf')
    
    for i, init in enumerate(top_inits):
        if time.time() >= deadline - 10:
            break
        b, bv = _sa_run(n_rows, k, init, budget_per_sa, seed=42 + i * 7)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    if overall_best is None:
        overall_best = inits[0]
        overall_best_val = _max_entry(overall_best)

    # Phase 3: Polish
    if time.time() < deadline - 5:
        polish_budget = deadline - time.time() - 2
        b, bv = _polish_solution(overall_best, n_rows, k, polish_budget, seed=999)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Final validation
    if not _valid(overall_best, n_rows, k):
        overall_best = _build_greedy(n_rows, k, rng=None)

    return overall_best

# EVOLVE-BLOCK-END
