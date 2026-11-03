"""Difference triangle set (7,5) solver - rewritten with structured search and aggressive local search."""
import time
import random
import math

_DEADLINE = 142.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    n = len(row)
    return [row[j] - row[i] for i in range(n) for j in range(i+1, n)]

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    if not rows:
        return False
    seen = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(k+1):
            for j in range(i+1, k+1):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _build_row_given_used(used, k, rng, cap=None, jitter=0.0):
    """Build a row of k+1 numbers starting at 0, avoiding used differences."""
    row = [0]
    x = 0
    attempts = 0
    while len(row) < k + 1:
        x += 1
        if cap is not None and x > cap:
            return None
        # Add randomness
        if jitter > 0 and rng.random() < jitter:
            x += rng.randint(1, 3)
            if cap is not None and x > cap:
                return None
        
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used = used | new_diffs
    return row

def _initial_construction(n_rows, k, rng, diversity_base=0.04):
    """Build initial valid set with varied diversity parameters."""
    best = None
    best_score = float('inf')
    
    for attempt in range(500):
        if _elapsed() > _DEADLINE - 25:
            break
        used = set()
        rows = []
        ok = True
        for r in range(n_rows):
            div = diversity_base * (0.5 + rng.random())
            row = _build_row_given_used(used, k, rng, jitter=div)
            if row is None:
                ok = False
                break
            rows.append(row)
            used = used | set(_row_diffs(row))
        
        if ok and len(rows) == n_rows:
            if _is_valid(rows, k):
                score = _max_entry(rows)
                if score < best_score:
                    best_score = score
                    best = [list(r) for r in rows]
    
    return best, best_score

def _compute_all_diffs(rows):
    """Return flat list of all differences."""
    all_d = []
    for row in rows:
        all_d.extend(_row_diffs(row))
    return all_d

def _count_conflicts(rows):
    """Count number of differences that appear more than once."""
    from collections import Counter
    all_d = _compute_all_diffs(rows)
    c = Counter(all_d)
    return sum(max(0, v - 1) for v in c.values())

def _sa_main(rows, k, rng, time_budget=100):
    """Simulated annealing with better move set."""
    n_rows = len(rows)
    cur = [list(r) for r in rows]
    best = [list(r) for r in rows]
    
    # Track diff usage for fast conflict counting
    from collections import Counter
    def diff_counter(rows):
        c = Counter()
        for row in rows:
            for d in _row_diffs(row):
                c[d] += 1
        return c
    
    def conflict_count(c):
        return sum(max(0, v - 1) for v in c.values())
    
    dc = diff_counter(cur)
    cur_conf = conflict_count(dc)
    cur_scope = _max_entry(cur)
    cur_cost = cur_scope + 500 * cur_conf
    
    best_scope = cur_scope
    best_conf = cur_conf
    best_cost = cur_cost
    best = [list(r) for r in cur]
    
    t0 = time.time()
    temp = 10.0
    cool = 0.99985
    
    while time.time() - t0 < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
        
        move_type = rng.random()
        ri = rng.randrange(n_rows)
        
        if move_type < 0.45:
            # Rebuild row ri
            old_row = cur[ri]
            old_diffs = _row_diffs(old_row)
            
            # Remove old diffs
            for d in old_diffs:
                dc[d] -= 1
                if dc[d] <= 0:
                    del dc[d]
            new_conf = conflict_count(dc)
            
            # Try to rebuild with tighter cap
            used = set(dc.keys())
            cap = cur_scope + rng.randint(0, 3)
            new_row = _build_row_given_used(used, k, rng, cap=cap, jitter=0.06)
            
            if new_row is None or len(new_row) < k + 1:
                # Restore
                for d in old_diffs:
                    dc[d] = dc.get(d, 0) + 1
                continue
            
            new_diffs = _row_diffs(new_row)
            for d in new_diffs:
                dc[d] = dc.get(d, 0) + 1
            new_conf = conflict_count(dc)
            new_scope = _max_entry(cur)
            new_cost = new_scope + 500 * new_conf
            delta = new_cost - cur_cost
            
            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                cur[ri] = new_row
                cur_conf = new_conf
                cur_scope = new_scope
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_scope = cur_scope
                    best_conf = cur_conf
                    best = [list(r) for r in cur]
            else:
                # Revert
                for d in new_diffs:
                    dc[d] -= 1
                    if dc[d] <= 0:
                        del dc[d]
                for d in old_diffs:
                    dc[d] = dc.get(d, 0) + 1
                cur[ri] = old_row
                cur_conf = new_conf  # will be recomputed properly
                # Recompute
                dc2 = diff_counter(cur)
                dc = dc2
                cur_conf = conflict_count(dc)
                cur_scope = _max_entry(cur)
                cur_cost = cur_scope + 500 * cur_conf
        
        elif move_type < 0.75:
            # Shift one entry
            row = cur[ri]
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            
            # Try to reduce
            if rng.random() < 0.65:
                new_val = old_val - rng.randint(1, 4)
            else:
                new_val = old_val + rng.randint(-3, 3)
            
            if new_val <= 0 or new_val == old_val:
                continue
            if idx > 1 and new_val <= row[idx-1]:
                continue
            if idx < len(row) - 1 and new_val >= row[idx+1]:
                continue
            
            old_diffs = _row_diffs(row)
            for d in old_diffs:
                dc[d] -= 1
                if dc[d] <= 0:
                    del dc[d]
            
            row[idx] = new_val
            new_diffs = _row_diffs(row)
            for d in new_diffs:
                dc[d] = dc.get(d, 0) + 1
            
            new_conf = conflict_count(dc)
            new_scope = _max_entry(cur)
            new_cost = new_scope + 500 * new_conf
            delta = new_cost - cur_cost
            
            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                cur_conf = new_conf
                cur_scope = new_scope
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_scope = cur_scope
                    best_conf = new_conf
                    best = [list(r) for r in cur]
            else:
                row[idx] = old_val
                for d in new_diffs:
                    dc[d] -= 1
                    if dc[d] <= 0:
                        del dc[d]
                for d in old_diffs:
                    dc[d] = dc.get(d, 0) + 1
        
        else:
            # Swap two entries within a row (maintaining order)
            row = cur[ri]
            if len(row) < 3:
                temp *= cool
                continue
            i = rng.randint(1, len(row) - 2)
            j = rng.randint(i + 1, len(row) - 1)
            if row[j] - row[i] < 2:
                temp *= cool
                continue
            # Try swapping is hard with ordering; instead, try a small perturbation of two entries
            di = rng.randint(-2, 2)
            dj = rng.randint(-2, 2)
            ni = row[i] + di
            nj = row[j] + dj
            if ni <= 0 or nj <= 0 or ni == row[i] and nj == row[j]:
                temp *= cool
                continue
            if i > 1 and ni <= row[i-1]:
                temp *= cool
                continue
            if j < len(row) - 1 and nj >= row[j+1]:
                temp *= cool
                continue
            if ni >= nj:
                temp *= cool
                continue
            
            old_diffs = _row_diffs(row)
            for d in old_diffs:
                dc[d] -= 1
                if dc[d] <= 0:
                    del dc[d]
            
            row[i] = ni
            row[j] = nj
            new_diffs = _row_diffs(row)
            for d in new_diffs:
                dc[d] = dc.get(d, 0) + 1
            
            new_conf = conflict_count(dc)
            new_scope = _max_entry(cur)
            new_cost = new_scope + 500 * new_conf
            delta = new_cost - cur_cost
            
            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                cur_conf = new_conf
                cur_scope = new_scope
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_scope = cur_scope
                    best_conf = new_conf
                    best = [list(r) for r in cur]
            else:
                row[i] = row[i] - di  # restore
                row[j] = row[j] - dj
                for d in new_diffs:
                    dc[d] -= 1
                    if dc[d] <= 0:
                        del dc[d]
                for d in old_diffs:
                    dc[d] = dc.get(d, 0) + 1
        
        temp = max(0.01, temp * cool)
    
    if _is_valid(best, k):
        return best
    if _is_valid(cur, k):
        return cur
    return best

def _aggressive_polish(rows, k, rng, time_budget=8):
    """Aggressive final polish: try many single-entry reductions."""
    best = [list(r) for r in rows]
    best_scope = _max_entry(best)
    
    t0 = time.time()
    improved = True
    while improved and time.time() - t0 < time_budget:
        if _elapsed() > _DEADLINE - 2:
            break
        improved = False
        all_diffs = set()
        for row in best:
            for d in _row_diffs(row):
                all_diffs.add(d)
        
        for ri in range(len(best)):
            if _elapsed() > _DEADLINE - 2:
                break
            row = best[ri]
            for idx in range(1, len(row)):
                if _elapsed() > _DEADLINE - 2:
                    break
                old_val = row[idx]
                for red in range(1, 8):
                    new_val = old_val - red
                    if new_val <= 0:
                        break
                    if idx > 1 and new_val <= row[idx-1]:
                        break
                    if idx < len(row) - 1 and new_val >= row[idx+1]:
                        break
                    
                    # Check conflicts
                    old_d = _row_diffs(row)
                    row[idx] = new_val
                    new_d = _row_diffs(row)
                    
                    temp_set = set(all_diffs)
                    for d in old_d:
                        temp_set.discard(d)
                    ok = True
                    for d in new_d:
                        if d in temp_set:
                            ok = False
                            break
                        temp_set.add(d)
                    
                    if ok:
                        all_diffs = temp_set
                        new_scope = _max_entry(best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            improved = True
                        break
                    else:
                        row[idx] = old_val
                        continue
                # Restore if we didn't break
                # (row[idx] is already correct)
    
    if _is_valid(best, k):
        return best
    return [list(r) for r in rows]

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(20250705)
    
    # Phase 1: Initial construction
    rows, score = _initial_construction(n_rows, k, rng)
    
    if rows is None:
        # Fallback greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
        score = _max_entry(rows)
    
    # Phase 2: SA optimization
    if _elapsed() < _DEADLINE - 12:
        tb = min(105, _DEADLINE - 12 - _elapsed())
        if tb > 5:
            opt = _sa_main(rows, k, rng, time_budget=tb)
            if opt and _is_valid(opt, k):
                ns = _max_entry(opt)
                if ns < score:
                    rows = opt
                    score = ns
    
    # Phase 3: Aggressive polish
    if _elapsed() < _DEADLINE - 2:
        tb = min(8, _DEADLINE - 2 - _elapsed())
        if tb > 0.5:
            pol = _aggressive_polish(rows, k, rng, time_budget=tb)
            if pol and _is_valid(pol, k):
                ns = _max_entry(pol)
                if ns < score:
                    rows = pol
                    score = ns
    
    # Final validation
    if not _is_valid(rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
    
    return rows
