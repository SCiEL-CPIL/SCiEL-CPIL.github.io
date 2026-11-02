"""Simulated-annealing search for difference triangle set (7,5) minimizing largest entry."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return set of all positive pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_valid_row(row):
    """Check a row is strictly increasing, starts at 0, and all diffs are distinct."""
    if not row or row[0] != 0:
        return False
    n = len(row)
    for i in range(1, n):
        if row[i] <= row[i - 1]:
            return False
    return len(_row_diffs(row)) == n * (n - 1) // 2

def _valid(rows, n_rows, k):
    """Check all rows valid and all differences globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if not _is_valid_row(row):
            return False
        d = _row_diffs(row)
        if d & used:
            return False
        used |= d
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_row(used, k, rng=None, max_val=5000):
    """Build a single row greedily avoiding used diffs. Returns row or None."""
    row = [0]
    while len(row) < k + 1:
        x = row[-1] + 1
        found = False
        while x <= max_val:
            new_diffs = [x - a for a in row]
            if not any(d in used for d in new_diffs) and len(set(new_diffs)) == len(new_diffs):
                row.append(x)
                used.update(new_diffs)
                found = True
                break
            x += 1
        if not found:
            return None
    return row

def _build_greedy(n_rows, k, rng=None):
    """Build all rows greedily."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng=rng)
        if row is None:
            return None
        rows.append(row)
    return rows

def _resolve_row(ri, rows, k, rng, max_target=None):
    """Rebuild row ri avoiding other rows' diffs."""
    other_used = set()
    for i, r in enumerate(rows):
        if i != ri:
            other_used |= _row_diffs(r)
    fresh_used = set(other_used)
    if max_target is None:
        max_target = max(rows[ri]) + 20 if rows[ri] else 200
    row = _greedy_row(fresh_used, k, rng=rng, max_val=max_target)
    return row

def _sa_move_single_pos(cur, ri, pi, rng, k):
    """Try moving position pi in row ri. Returns (new_row, delta_max) or (None, 0)."""
    row = cur[ri]
    old_val = row[pi]
    lo = row[pi - 1] + 1
    hi = row[pi + 1] - 1 if pi < k else old_val + 50
    if hi < lo:
        return None, 0
    
    # Generate candidates
    candidates = set()
    span = max(1, hi - lo + 1)
    for _ in range(12):
        nv = rng.randint(lo, hi)
        candidates.add(nv)
    for d in (-3, -2, -1, 1, 2, 3, 5, -5):
        nv = old_val + d
        if lo <= nv <= hi:
            candidates.add(nv)
    candidates.discard(old_val)
    if not candidates:
        return None, 0
    
    for nv in rng.sample(list(candidates), min(3, len(candidates))):
        cand_row = row[:]
        cand_row[pi] = nv
        if not _is_valid_row(cand_row):
            continue
        new_diffs = _row_diffs(cand_row)
        old_diffs = _row_diffs(row)
        # Check no collision with other rows
        other_diffs = set()
        for i, r in enumerate(cur):
            if i != ri:
                other_diffs |= _row_diffs(r)
        if new_diffs & other_diffs:
            continue
        # Valid move
        old_max = _max_entry(cur)
        new_max = max(max(cand_row), max(max(r) for i, r in enumerate(cur) if i != ri))
        delta = new_max - old_max
        return cand_row, delta
    return None, 0

def _sa_move_swap(cur, ri, rj, rng, k):
    """Swap rows ri and rj. Returns (new_ri, new_rj, delta) or None."""
    if ri == rj:
        return None
    old_max = _max_entry(cur)
    new_ri = cur[rj][:]
    new_rj = cur[ri][:]
    new_max = max(max(new_ri), max(new_rj), max(max(r) for i, r in enumerate(cur) if i != ri and i != rj))
    delta = new_max - old_max
    return new_ri, new_rj, delta

def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run SA from initial_rows. Returns (best_rows, best_val)."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    
    T = 8.0
    T_end = 0.1
    decay = 0.9997
    t0 = time.time()
    steps = 0
    
    while time.time() - t0 < time_budget:
        steps += 1
        r = rng.random()
        
        if r < 0.50:
            # Single position move
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, ri, pi, rng, k)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / max(T, T_end)):
                    cur[ri] = new_row
                    cur_val = _max_entry(cur)
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
        elif r < 0.70:
            # Row swap
            if n_rows >= 2:
                ri = rng.randrange(n_rows)
                rj = rng.randrange(n_rows)
                if ri != rj:
                    result = _sa_move_swap(cur, ri, rj, rng, k)
                    if result is not None:
                        new_ri, new_rj, delta = result
                        if delta <= 0 or rng.random() < math.exp(-delta / max(T, T_end)):
                            cur[ri], cur[rj] = new_ri, new_rj
                            cur_val = _max_entry(cur)
                            if cur_val < best_val:
                                best_val = cur_val
                                best = [r2[:] for r2 in cur]
        elif r < 0.88:
            # Re-resolve a row
            ri = rng.randrange(n_rows)
            new_row = _resolve_row(ri, cur, k, rng, max_target=max(cur[ri]) + 15)
            if new_row is not None:
                # Verify no collision
                other_used = set()
                for i, r in enumerate(cur):
                    if i != ri:
                        other_used |= _row_diffs(r)
                if not (_row_diffs(new_row) & other_used):
                    old_max = _max_entry(cur)
                    cur[ri] = new_row
                    new_max = _max_entry(cur)
                    delta = new_max - old_max
                    if delta <= 0 or rng.random() < math.exp(-delta / max(T, T_end)):
                        cur_val = new_max
                        if cur_val < best_val:
                            best_val = cur_val
                            best = [r2[:] for r2 in cur]
        else:
            # Kick: perturb from best
            cur = [r2[:] for r2 in best]
            cur_val = best_val
            for _ in range(rng.randint(1, 2)):
                ri = rng.randrange(n_rows)
                pi = rng.randrange(1, k + 1)
                row = cur[ri]
                lo = row[pi - 1] + 1
                hi = row[pi + 1] - 1 if pi < k else row[pi] + 30
                if hi >= lo:
                    nv = rng.randint(lo, hi)
                    cand_row = row[:]
                    cand_row[pi] = nv
                    if _is_valid_row(cand_row):
                        other_used = set()
                        for i, r in enumerate(cur):
                            if i != ri:
                                other_used |= _row_diffs(r)
                        if not (_row_diffs(cand_row) & other_used):
                            cur[ri] = cand_row
                            cur_val = _max_entry(cur)
        
        T = max(T_end, T * decay)
        
        # Periodic re-anchor
        if steps % 1500 == 0 and cur_val > best_val + 12:
            cur = [r2[:] for r2 in best]
            cur_val = best_val
    
    return best, best_val

def _polish(rows, n_rows, k, time_budget, seed):
    """Final high-precision polish: try to reduce max by 1 via targeted search."""
    rng = random.Random(seed)
    cur = [r[:] for r in rows]
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    t0 = time.time()
    
    # Try to find a configuration with max = cur_val - 1
    target = cur_val - 1
    while time.time() - t0 < time_budget and best_val > target:
        # Try resolving each row with tighter bound
        for ri in range(n_rows):
            if time.time() - t0 > time_budget:
                break
            new_row = _resolve_row(ri, cur, k, rng, max_target=target)
            if new_row is not None:
                other_used = set()
                for i, r in enumerate(cur):
                    if i != ri:
                        other_used |= _row_diffs(r)
                if not (_row_diffs(new_row) & other_used):
                    cur[ri] = new_row
                    new_val = _max_entry(cur)
                    if new_val < best_val:
                        best_val = new_val
                        best = [r2[:] for r2 in cur]
                        target = best_val - 1
    
    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Phase 1: diverse initializations
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(10):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(5000 + s)))
    inits = [x for x in inits if x is not None]
    if not inits:
        # Fallback: simple sequential
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = [x - a for a in row]
                if not any(d in used for d in new_diffs) and len(set(new_diffs)) == len(new_diffs):
                    row.append(x)
                    used.update(new_diffs)
                x += 1
            rows.append(row)
        return rows
    
    inits.sort(key=_max_entry)
    top_inits = inits[:5]
    
    # Phase 2: SA trajectories
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    n_sa = min(5, len(top_inits))
    budget_per_sa = max(3.0, (remaining - 10) / n_sa)
    
    overall_best = top_inits[0]
    overall_best_val = _max_entry(overall_best)
    
    for i in range(n_sa):
        if time.time() >= deadline - 12:
            break
        b, bv = _sa_run(n_rows, k, top_inits[i], budget_per_sa, seed=100 + i * 13)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b
    
    # Phase 3: polish
    if time.time() < deadline - 5:
        b, bv = _polish(overall_best, n_rows, k, max(3.0, deadline - time.time() - 3), seed=777)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b
    
    # Final validation
    if not _valid(overall_best, n_rows, k):
        overall_best = _build_greedy(n_rows, k, rng=None)
        if overall_best is None:
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new_diffs = [x - a for a in row]
                    if not any(d in used for d in new_diffs) and len(set(new_diffs)) == len(new_diffs):
                        row.append(x)
                        used.update(new_diffs)
                    x += 1
                rows.append(row)
            overall_best = rows
    
    return overall_best
# EVOLVE-BLOCK-END
