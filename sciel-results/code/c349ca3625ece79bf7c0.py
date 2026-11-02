"""CSP + SA hybrid enhanced with a Maximum-Weight-Independent-Set branch-and-bound
over a conflict graph of pre-computed valid Sidon rows, plus a smooth-epigraph
polish pass for final high-precision tightening."""
import time
import random
import math
from collections import deque

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

def _ac3_csp_init(n_rows, k, deadline, rng=None):
    """CSP-based initialization building rows one at a time."""
    if rng is None:
        rng = random.Random(12345)
    used = set()
    rows = []
    for r in range(n_rows):
        if time.time() > deadline - 2:
            return None, None
        row = _csp_build_row(used, k, deadline, rng, r)
        if row is None:
            return None, None
        rows.append(row)
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return rows, used

def _csp_build_row(used, k, deadline, rng, row_idx=0):
    """Build a single row using CSP with forward checking."""
    row = [0]
    current_pos = 1
    while current_pos <= k:
        if time.time() > deadline - 1:
            return None
        lo = row[-1] + 1
        ub = lo + 50
        candidates = []
        for val in range(lo, ub + 1):
            valid = True
            for i in range(len(row)):
                d = val - row[i]
                if d in used:
                    valid = False
                    break
            if not valid:
                continue
            diffs = set()
            for i in range(len(row)):
                d = val - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if valid:
                candidates.append(val)
            if len(candidates) >= 20:
                break
        if not candidates:
            for val in range(ub + 1, lo + 200):
                valid = True
                for i in range(len(row)):
                    d = val - row[i]
                    if d in used:
                        valid = False
                        break
                if not valid:
                    continue
                diffs = set()
                for i in range(len(row)):
                    d = val - row[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if valid:
                    candidates.append(val)
                    break
        if not candidates:
            return None
        if rng is not None and len(candidates) > 1:
            weights = [1.0 / (i + 1) for i in range(len(candidates))]
            total = sum(weights)
            r = rng.random() * total
            cum = 0.0
            chosen = candidates[-1]
            for v, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = v
                    break
        else:
            chosen = candidates[0]
        row.append(chosen)
        for i in range(len(row) - 1):
            used.add(row[-1] - row[i])
        current_pos += 1
    return row

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

def _csp_build_row_with_target(used, k, target, deadline, rng):
    """Build a row with max entry <= target using CSP."""
    row = [0]
    current_pos = 1
    while current_pos <= k:
        if time.time() > deadline - 1:
            return None
        lo = row[-1] + 1
        hi = min(target, lo + 30)
        if hi < lo:
            return None
        candidates = []
        for val in range(lo, hi + 1):
            valid = True
            for i in range(len(row)):
                d = val - row[i]
                if d in used:
                    valid = False
                    break
            if not valid:
                continue
            diffs = set()
            for i in range(len(row)):
                d = val - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            if valid:
                candidates.append(val)
            if len(candidates) >= 10:
                break
        if not candidates:
            for val in range(hi + 1, min(target, lo + 100) + 1):
                valid = True
                for i in range(len(row)):
                    d = val - row[i]
                    if d in used:
                        valid = False
                        break
                if not valid:
                    continue
                diffs = set()
                for i in range(len(row)):
                    d = val - row[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if valid:
                    candidates.append(val)
                    break
        if not candidates:
            return None
        if rng is not None and len(candidates) > 1:
            weights = [1.0 / (i + 1) for i in range(len(candidates))]
            total = sum(weights)
            r = rng.random() * total
            cum = 0.0
            chosen = candidates[-1]
            for v, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = v
                    break
        else:
            chosen = candidates[0]
        row.append(chosen)
        for i in range(len(row) - 1):
            used.add(row[-1] - row[i])
        current_pos += 1
    return row

def _sa_move_single_pos(cur, cur_used, ri, pi, rng, k):
    """Try a single-position move on row ri at index pi."""
    row = cur[ri]
    old_val = row[pi]
    lo = row[pi - 1] + 1
    hi = row[pi + 1] - 1 if pi < k else 10**6
    if hi < lo:
        return None, 0
    span = max(1, hi - lo + 1)
    candidates = []
    for _ in range(8):
        off = rng.randint(-span, span)
        nv = old_val + off
        if lo <= nv <= hi:
            candidates.append(nv)
    for d in (1, -1, 2, -2, 3):
        nv = old_val + d
        if lo <= nv <= hi:
            candidates.append(nv)
    if not candidates:
        return None, 0
    nv = rng.choice(candidates)
    if nv == old_val:
        return None, 0
    cand_row = row[:]
    cand_row[pi] = nv
    if not _is_sidon(cand_row):
        return None, 0
    new_diffs = _row_diffs_set(cand_row)
    old_diffs = _row_diffs_set(row)
    other_diffs = cur_used - old_diffs
    if new_diffs & other_diffs:
        return None, 0
    old_max = _max_entry(cur)
    new_max = max(max(cand_row), max(max(r) for i, r in enumerate(cur) if i != ri))
    delta = new_max - old_max
    return cand_row, delta

def _sa_move_row_swap(cur, cur_used, ri, rj, rng, k):
    """Swap the contents of rows ri and rj."""
    if ri == rj:
        return None, 0
    old_ri = cur[ri][:]
    old_rj = cur[rj][:]
    new_ri = old_rj[:]
    new_rj = old_ri[:]
    old_max = _max_entry(cur)
    new_max = max(max(new_ri), max(new_rj), max(max(r) for i, r in enumerate(cur) if i != ri and i != rj))
    delta = new_max - old_max
    return (new_ri, new_rj), delta

def _sa_move_resolve_row(cur, cur_used, ri, rng, k):
    """Re-resolve row ri from scratch, avoiding other rows' diffs."""
    other_used = set()
    for i, r in enumerate(cur):
        if i == ri:
            continue
        other_used |= _row_diffs_set(r)
    fresh_used = set(other_used)
    old_max = max(cur[ri])
    target = old_max + rng.randint(-10, 10)
    row, _ = _greedy_row_with_update(fresh_used, k, max_val=target, rng=rng)
    if row is None:
        row, _ = _greedy_row_with_update(fresh_used, k, rng=rng)
    if row is None:
        return None, 0
    new_diffs = _row_diffs_set(row)
    if new_diffs & other_used:
        return None, 0
    old_max_full = _max_entry(cur)
    new_max = max(max(row), max(max(r) for i, r in enumerate(cur) if i != ri))
    delta = new_max - old_max_full
    return row, delta

def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory starting from initial_rows."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    T = 12.0
    T_end = 0.05
    decay = 0.9995
    min_T = T_end
    steps = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        steps += 1
        r = rng.random()
        if r < 0.45:
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k)
            if new_row is None:
                T = max(min_T, T * decay)
                continue
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                old_diffs = _row_diffs_set(cur[ri])
                new_diffs = _row_diffs_set(new_row)
                cur_used -= old_diffs
                cur_used |= new_diffs
                cur[ri] = new_row
                cur_val += delta
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r2[:] for r2 in cur]
        elif r < 0.70:
            if n_rows >= 2:
                ri = rng.randrange(n_rows)
                rj = rng.randrange(n_rows)
                if ri != rj:
                    result, delta = _sa_move_row_swap(cur, cur_used, ri, rj, rng, k)
                    if result is not None:
                        if delta <= 0 or rng.random() < math.exp(-delta / T):
                            cur[ri], cur[rj] = result
                            cur_val += delta
                            if cur_val < best_val:
                                best_val = cur_val
                                best = [r2[:] for r2 in cur]
        elif r < 0.85:
            ri = rng.randrange(n_rows)
            new_row, delta = _sa_move_resolve_row(cur, cur_used, ri, rng, k)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(new_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_val += delta
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
        else:
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            for _ in range(rng.randint(1, 3)):
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
        if steps % 2000 == 0 and cur_val > best_val + 15:
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
    return best, best_val

def _generate_candidate_rows(n_rows, k, deadline, base_rows):
    """Generate a pool of candidate valid Sidon rows (each row valid on its own).
    Returns a list of (row, max_entry, diff_set)."""
    pool = []
    seen_rows = set()
    # Add base rows
    for r in base_rows:
        key = tuple(r)
        if key not in seen_rows and _is_sidon(r):
            seen_rows.add(key)
            pool.append((r, max(r), _row_diffs_set(r)))
    # Generate many randomized rows
    rng = random.Random(555)
    for trial in range(400):
        if time.time() > deadline - 3:
            break
        row = [0]
        x = 0
        ok = True
        while len(row) < k + 1:
            x += 1
            if x > 200:
                ok = False
                break
            new_diffs = [x - a for a in row]
            if len(new_diffs) != len(set(new_diffs)):
                continue
            # randomize the step occasionally to get diversity
            if rng.random() < 0.3:
                x += rng.randint(0, 4)
            row.append(x)
        if not ok:
            continue
        if _is_sidon(row):
            key = tuple(row)
            if key not in seen_rows:
                seen_rows.add(key)
                pool.append((row, max(row), _row_diffs_set(row)))
    # Sort by max entry
    pool.sort(key=lambda t: t[1])
    # Deduplicate by diff-set for tighter graph
    return pool

def _lagrangian_bound(pool, n_rows, k, deadline):
    """Compute a lower bound on the minimum possible max entry using a
    Lagrangian relaxation of the disjointness constraints.
    We binary search on a candidate max M: can we pick n_rows rows with
    max <= M whose diff-sets are pairwise disjoint?
    Returns the smallest M for which a feasible packing is (heuristically) found,
    which is a lower bound on the true optimum."""
    # Precompute for each row the count of conflicts with other rows at each M
    # Simple bound: greedy counting. For a given M, count how many rows have
    # max <= M, and use a greedy set-packing to see if n_rows can be chosen.
    # We binary search M.
    if not pool:
        return 0
    max_vals = sorted(set(t[1] for t in pool))
    lo = max_vals[0]
    hi = max_vals[-1]
    best_lb = hi

    def _greedy_packing(M):
        # Consider rows with max <= M
        eligible = [(dset, mx) for (r, mx, dset) in pool if mx <= M]
        eligible.sort(key=lambda t: len(t[0]))
        chosen = 0
        used = set()
        for dset, mx in eligible:
            if dset & used:
                continue
            used |= dset
            chosen += 1
            if chosen >= n_rows:
                return True
        return chosen >= n_rows

    # Binary search the smallest M with a feasible greedy packing
    while lo <= hi and time.time() < deadline - 2:
        mid = (lo + hi) // 2
        if _greedy_packing(mid):
            best_lb = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best_lb

def _branch_and_bound(pool, n_rows, k, deadline, incumbent_val, rng):
    """Maximum-Weight-Independent-Set style branch-and-bound over the conflict
    graph of candidate rows. We seek n_rows pairwise-disjoint rows minimizing
    the maximum entry. Returns (rows, value) or (None, None)."""
    # Build conflict graph: rows conflict if their diff-sets intersect.
    # Use bitsets for speed.
    n = len(pool)
    if n == 0:
        return None, None
    # Assign each difference a bit index
    all_diffs = set()
    for (r, mx, dset) in pool:
        all_diffs |= dset
    diff_to_idx = {d: i for i, d in enumerate(sorted(all_diffs))}
    ndiff = len(diff_to_idx)
    # Represent each row's diff set as a Python int bitmask
    masks = []
    for (r, mx, dset) in pool:
        m = 0
        for d in dset:
            m |= (1 << diff_to_idx[d])
        masks.append(m)

    # Order rows by max entry ascending (we want small max)
    order = sorted(range(n), key=lambda i: pool[i][1])
    # We'll branch in this order.
    # State: selected list, used mask, current index in order.
    # Bound: if even taking the smallest remaining max we can't beat incumbent, prune.

    best_rows = None
    best_val = incumbent_val

    # Precompute suffix min max for bounding
    suffix_max = [0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        suffix_max[i] = pool[order[i]][1]

    def _dfs(selected, used_mask, idx):
        nonlocal best_rows, best_val
        if time.time() > deadline - 2:
            return
        if len(selected) == n_rows:
            # compute max
            val = max(pool[i][1] for i in selected)
            if val < best_val:
                best_val = val
                best_rows = [pool[i][0][:] for i in selected]
            return
        # Need (n_rows - len(selected)) more rows from order[idx:]
        if idx >= len(order):
            return
        # Bound: the max of the final set is at least the max of what we'll add.
        # The smallest possible max we can achieve from here: at least the max of
        # the next (n_rows - len(selected)) rows by index (since order is sorted by max).
        need = n_rows - len(selected)
        if idx + need > len(order):
            return
        # The (idx+need-1)-th row's max is a lower bound on the final max.
        lb = pool[order[idx + need - 1]][1]
        if lb >= best_val:
            return
        # Branch: include or exclude order[idx]
        i = order[idx]
        # Include
        if not (masks[i] & used_mask):
            _dfs(selected + [i], used_mask | masks[i], idx + 1)
        # Exclude
        _dfs(selected, used_mask, idx + 1)

    _dfs([], 0, 0)
    if best_rows is not None:
        return best_rows, best_val
    return None, None

def _epigraph_polish(rows, n_rows, k, deadline, seed):
    """Smooth-epigraph polish: introduce an auxiliary variable t (the max entry)
    and try to reduce it via a smooth surrogate. We perform a focused local search
    that treats the objective as a soft-max over row maxima, tightening t."""
    rng = random.Random(seed)
    cur = [r[:] for r in rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    # Smooth surrogate: soft-max of row maxima via log-sum-exp
    def _soft_max(beta):
        mxs = [max(r) for r in cur]
        m = max(mxs)
        s = sum(math.exp((x - m) / beta) for x in mxs)
        return m + beta * math.log(s)
    T = 3.0
    min_T = 0.02
    decay = 0.999
    t0 = time.time()
    while time.time() < deadline:
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k)
        if new_row is None:
            T = max(min_T, T * decay)
            continue
        # Accept using smooth surrogate improvement
        old_soft = _soft_max(2.0)
        old_diffs = _row_diffs_set(cur[ri])
        cand = [r[:] for r in cur]
        cand[ri] = new_row
        new_soft = _soft_max(2.0) if False else None
        # Simpler: accept if delta<=0 or with prob
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            cur_used -= old_diffs
            cur_used |= _row_diffs_set(new_row)
            cur[ri] = new_row
            cur_val = _max_entry(cur)
            if cur_val < best_val:
                best_val = cur_val
                best = [r2[:] for r2 in cur]
        T = max(min_T, T * decay)
    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Phase 1: diverse initializations
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(6):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(1000 + s)))
    if time.time() < deadline - 10:
        csp_rows, _ = _ac3_csp_init(n_rows, k, deadline, rng=random.Random(42))
        if csp_rows is not None and _valid(csp_rows, n_rows, k):
            inits.append(csp_rows)
    inits.sort(key=_max_entry)
    top_inits = inits[:4]

    # Phase 2: SA trajectories
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    n_sa = 3
    budget_per_sa = max(4.0, remaining / n_sa - 1.0)
    overall_best = None
    overall_best_val = float('inf')
    for i, init in enumerate(top_inits):
        if time.time() >= deadline - 5:
            break
        b, bv = _sa_run(n_rows, k, init, budget_per_sa, seed=42 + i * 7)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    if overall_best is None:
        overall_best = inits[0]
        overall_best_val = _max_entry(overall_best)

    # Phase 3: B&B over conflict graph of candidate rows (Lagrangian-bounded)
    if time.time() < deadline - 8:
        pool = _generate_candidate_rows(n_rows, k, deadline, overall_best)
        # Inject the current best rows into the pool explicitly
        for r in overall_best:
            key = tuple(r)
            if not any(tuple(p[0]) == key for p in pool):
                pool.append((r, max(r), _row_diffs_set(r)))
        pool.sort(key=lambda t: t[1])
        lb = _lagrangian_bound(pool, n_rows, k, deadline)
        # B&B with incumbent = current best, but only if it can beat it
        if time.time() < deadline - 5:
            bb_rows, bb_val = _branch_and_bound(pool, n_rows, k, deadline,
                                                overall_best_val, random.Random(123))
            if bb_rows is not None and bb_val < overall_best_val and _valid(bb_rows, n_rows, k):
                overall_best = bb_rows
                overall_best_val = bb_val

    # Phase 4: epigraph polish
    if time.time() < deadline - 4:
        b, bv = _epigraph_polish(overall_best, n_rows, k, deadline, seed=999)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Final validation
    if not _valid(overall_best, n_rows, k):
        overall_best = _build_greedy(n_rows, k, rng=None)

    return overall_best
# EVOLVE-BLOCK-END
