"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _row_diffs(row):
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_sidon(row):
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return len(diffs) == len(set(diffs))

def _quick_greedy(n_rows, k, scope, rng):
    """Greedy construction with given RNG for tie-breaking."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for pos in range(1, k + 1):
            found = False
            candidates = []
            for v in range(row[-1] + 1, scope + 1):
                new_diffs = set()
                valid = True
                for a in row:
                    d = v - a
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                if new_diffs & used:
                    continue
                candidates.append((v, new_diffs))
                if len(candidates) >= 20:
                    break
            if not candidates:
                return None
            if len(candidates) > 1:
                idx = rng.randrange(min(10, len(candidates)))
            else:
                idx = 0
            v, nd = candidates[idx]
            row.append(v)
            used |= nd
        rows.append(row)
    return rows

def _try_swap_improve(n_rows, k, rows, scope, deadline, rng):
    """Try to improve by replacing individual rows with smaller-max alternatives."""
    cur = [r[:] for r in rows]
    cur_diffs = [_row_diffs(r) for r in cur]
    cur_max = max(r[-1] for r in cur)
    
    best = [r[:] for r in cur]
    best_max = cur_max
    no_improve = 0
    
    # Precompute: for each row, find alternative rows that fit
    # We'll do this on-the-fly to save memory
    
    while time.time() < deadline and no_improve < 30000:
        ri = rng.randrange(n_rows)
        old_row = cur[ri]
        old_max = old_row[-1]
        
        other_diffs = set()
        for j in range(n_rows):
            if j != ri:
                other_diffs |= cur_diffs[j]
        
        # Try to find a row with smaller max
        best_cand = None
        best_cand_max = old_max
        for v_last in range(1, old_max):
            # Try to build a row ending at v_last
            # This is expensive; instead, try random constructions
            pass
        
        # Simpler: try random perturbations of the row
        improved = False
        for _ in range(5):
            # Try to rebuild this row with a smaller max
            new_row = [0]
            target_max = old_max - rng.randrange(1, 5)
            if target_max <= new_row[-1]:
                continue
            for pos in range(1, k + 1):
                found = False
                candidates = []
                for v in range(new_row[-1] + 1, min(target_max, scope) + 1):
                    new_diffs = set()
                    valid = True
                    for a in new_row:
                        d = v - a
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    if new_diffs & other_diffs:
                        continue
                    candidates.append((v, new_diffs))
                    if len(candidates) >= 10:
                        break
                if not candidates:
                    break
                idx = rng.randrange(len(candidates))
                v, nd = candidates[idx]
                new_row.append(v)
            else:
                if len(new_row) == k + 1 and new_row[-1] < old_max:
                    cur[ri] = new_row
                    cur_diffs[ri] = _row_diffs(new_row)
                    new_total_max = max(r[-1] for r in cur)
                    if new_total_max < best_max:
                        best_max = new_total_max
                        best = [r[:] for r in cur]
                        no_improve = 0
                        improved = True
                        break
                    else:
                        no_improve += 1
                        # Keep the change for diversity
                        break
        
        if not improved:
            no_improve += 1
    
    return best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    # Lower bound on scope
    min_scope = n_rows * (k + 1) * k // 2
    
    # Phase 1: Quick greedy with multiple seeds to find initial solution
    best_result = None
    best_scope = None
    
    for scope in range(min_scope, min_scope + 40):
        if time.time() > deadline - 30:
            break
        found = False
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            rng = random.Random(seed)
            result = _quick_greedy(n_rows, k, scope, rng)
            if result is not None:
                if best_result is None:
                    best_result = result
                    best_scope = scope
                    found = True
                break
        if found:
            break
    
    if best_result is None:
        # Fallback with large scope
        rng = random.Random(0)
        best_result = _quick_greedy(n_rows, k, 500, rng)
        if best_result is None:
            # Emergency construction
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                v = 1
                while len(row) < k + 1:
                    ok = True
                    nd = set()
                    for a in row:
                        d = v - a
                        if d in nd or d in used:
                            ok = False
                            break
                        nd.add(d)
                    if ok:
                        row.append(v)
                        used |= nd
                    v += 1
                rows.append(row)
            return rows
        best_scope = max(r[-1] for r in best_result)
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 10:
        rng = random.Random(123)
        improved = _try_swap_improve(n_rows, k, best_result, best_scope, deadline - 5, rng)
        if improved is not None:
            if max(r[-1] for r in improved) < max(r[-1] for r in best_result):
                best_result = improved
    
    # Phase 3: More multi-start greedy with tighter scopes
    if time.time() < deadline - 10:
        for scope in range(min_scope, best_scope + 1):
            if time.time() > deadline - 5:
                break
            for seed in range(20):
                if time.time() > deadline - 5:
                    break
                rng = random.Random(seed * 1000 + scope)
                result = _quick_greedy(n_rows, k, scope, rng)
                if result is not None:
                    rmax = max(r[-1] for r in result)
                    if rmax < max(r[-1] for r in best_result):
                        best_result = result
                        best_scope = rmax
    
    return best_result
# EVOLVE-BLOCK-END
