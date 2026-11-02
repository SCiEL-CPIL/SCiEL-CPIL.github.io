"""CSP + SA hybrid: AC-3 constraint propagation for initialization and repair, combined with simulated annealing for global optimization."""
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
    """Build a single row greedily (optionally randomized), updating used as we go.
    If rng is None, deterministic smallest-first. Otherwise, pick from a small
    window of valid candidates with bias toward smaller values."""
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
            # Find a small window of valid candidates starting from x+1
            valid_xs = []
            scan = x + 1
            while scan <= min(x + 40, limit) and len(valid_xs) < 6:
                new_diffs = [scan - a for a in row]
                if not any(d in used for d in new_diffs) and len(new_diffs) == len(set(new_diffs)):
                    valid_xs.append(scan)
                scan += 1
            if not valid_xs:
                # Fallback: scan further
                scan = x + 1
                while scan <= limit:
                    new_diffs = [scan - a for a in row]
                    if not any(d in used for d in new_diffs) and len(new_diffs) == len(set(new_diffs)):
                        valid_xs.append(scan)
                        break
                    scan += 1
                if not valid_xs:
                    return None, used
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
    return row, used

def _ac3_csp_init(n_rows, k, deadline, rng=None):
    """CSP-based initialization using AC-3 arc consistency.
    Variables: (row, pos) for row in range(n_rows), pos in range(1, k+1)
    Domain: possible values for each variable
    Constraints: 
      1. Within a row: strictly increasing
      2. All pairwise differences globally unique
    
    Uses MRV heuristic to order variable assignments.
    Returns (rows, used_set) or (None, None) on failure."""
    if rng is None:
        rng = random.Random(12345)
    
    # Build initial domains
    # For position pos in row r, value must be > value at pos-1 in same row
    # Start with wide domains, then propagate
    
    # We'll build rows one at a time using CSP-style backtracking with MRV
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
    """Build a single row using CSP with MRV and forward checking."""
    row = [0]
    
    def _check_and_propagate(row_so_far, pos):
        """Check if adding a value at position pos is consistent."""
        if time.time() > deadline - 1:
            return False
        # Check all new differences are valid
        for i in range(len(row_so_far)):
            d = row_so_far[pos] - row_so_far[i]
            if d in used:
                return False
        # Check internal consistency (no duplicate diffs within row)
        diffs = set()
        for i in range(len(row_so_far)):
            d = row_so_far[pos] - row_so_far[i]
            if d in diffs:
                return False
            diffs.add(d)
        return True
    
    # MRV: at each step, find the position with fewest valid candidates
    # For simplicity, build left-to-right but use forward checking
    
    current_pos = 1
    while current_pos <= k:
        if time.time() > deadline - 1:
            return None
        
        # Find valid candidates for this position
        lo = row[-1] + 1
        # Upper bound: estimate based on remaining positions
        # Each remaining position needs at least +1, and diffs must be unique
        # Conservative upper bound
        ub = lo + 50  # scan window
        
        candidates = []
        for val in range(lo, ub + 1):
            test_row = row + [val]
            if _check_and_propagate(test_row, len(test_row) - 1):
                candidates.append(val)
            if len(candidates) >= 20:
                break
        
        if not candidates:
            # Expand search
            for val in range(ub + 1, lo + 200):
                test_row = row + [val]
                if _check_and_propagate(test_row, len(test_row) - 1):
                    candidates.append(val)
                    break
        
        if not candidates:
            return None
        
        # MRV: pick the candidate that leaves the most options for next position
        # Simplified: pick with some randomness biased toward smaller
        if rng is not None and len(candidates) > 1:
            # Weighted selection
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
        # Update used set for this row's internal diffs
        for i in range(len(row) - 1):
            d = row[-1] - row[i]
            used.add(d)
        
        current_pos += 1
    
    return row

def _build_greedy(n_rows, k, rng=None):
    """Fast greedy: each row takes smallest (or random-window) next mark with no diff collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = _greedy_row_with_update(used, k, rng=rng)
        if row is None:
            # Deterministic fallback with no limit
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

def _csp_repair_row(ri, rows, k, deadline, rng):
    """CSP-based repair of a single row, trying to minimize its max entry."""
    other_used = set()
    for i, r in enumerate(rows):
        if i == ri:
            continue
        other_used |= _row_diffs_set(r)
    
    old_row = rows[ri]
    old_max = max(old_row)
    
    # Try to rebuild with a target slightly below old_max
    target = old_max - 1
    for attempt in range(5):
        if time.time() > deadline - 1:
            break
        fresh_used = set(other_used)
        row = _csp_build_row_with_target(fresh_used, k, target, deadline, rng)
        if row is not None:
            new_diffs = _row_diffs_set(row)
            if not (new_diffs & other_used):
                return row
        target += 1
    
    return None

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
            test_row = row + [val]
            # Check all new diffs
            valid = True
            for i in range(len(row)):
                d = val - row[i]
                if d in used:
                    valid = False
                    break
            if not valid:
                continue
            # Check internal uniqueness
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
            # Try expanding
            for val in range(hi + 1, min(target, lo + 100) + 1):
                test_row = row + [val]
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
        
        # Pick with bias toward smaller
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

def _resolve_row_given_others(ri, rows, k, max_target, rng=None):
    """Rebuild row ri greedily, avoiding only other rows' diffs.
    Returns new row (max <= max_target) or None."""
    other_used = set()
    for i, r in enumerate(rows):
        if i == ri:
            continue
        other_used |= _row_diffs_set(r)
    # Build fresh with a fresh used that only contains other rows' diffs
    fresh_used = set(other_used)
    row, _ = _greedy_row_with_update(fresh_used, k, max_val=max_target, rng=rng)
    return row

def _sa_move_single_pos(cur, cur_used, ri, pi, rng, k, cur_val=None):
    """Try a single-position move on row ri at index pi.
    Returns (new_row, delta) where delta = new_max - old_max (can be negative, 0, or positive).
    If no valid move found, returns (None, 0)."""
    row = cur[ri]
    old_val = row[pi]
    lo = row[pi - 1] + 1
    hi = row[pi + 1] - 1 if pi < k else 10**6
    if hi < lo:
        return None, 0
    # Candidate range: within [lo, hi], but also allow some expansion
    span = max(1, hi - lo + 1)
    # Try a few random offsets
    candidates = []
    for _ in range(8):
        off = rng.randint(-span, span)
        nv = old_val + off
        if lo <= nv <= hi:
            candidates.append(nv)
    # Also try a few deterministic near moves
    for d in (1, -1, 2, -2, 3):
        nv = old_val + d
        if lo <= nv <= hi:
            candidates.append(nv)
    if not candidates:
        return None, 0
    # Pick a random candidate
    nv = rng.choice(candidates)
    if nv == old_val:
        return None, 0
    # Fast incremental validity check: only check new diffs against old row's diffs and cur_used
    # New diffs are nv - row[i] for i in range(pi) and row[j] - nv for j in range(pi+1, k+1)
    # Old diffs involving position pi are old_val - row[i] for i < pi and row[j] - old_val for j > pi
    # We need: (1) new diffs not in cur_used minus old diffs, (2) new diffs all distinct from each other
    # Compute new diffs
    new_diffs = set()
    for i in range(pi):
        d = nv - row[i]
        if d in new_diffs:
            return None, 0
        new_diffs.add(d)
    for j in range(pi + 1, k + 1):
        d = row[j] - nv
        if d in new_diffs:
            return None, 0
        new_diffs.add(d)
    # Old diffs that are being removed
    old_diffs_pi = set()
    for i in range(pi):
        old_diffs_pi.add(old_val - row[i])
    for j in range(pi + 1, k + 1):
        old_diffs_pi.add(row[j] - old_val)
    # Check new diffs against remaining used diffs
    other_diffs = cur_used - old_diffs_pi
    if new_diffs & other_diffs:
        return None, 0
    # Compute delta in objective (max entry)
    if cur_val is None:
        cur_val = _max_entry(cur)
    new_row_max = max(row)
    # The max of the new row is max(row with position pi replaced by nv)
    # row is increasing, so max(row) = row[k]
    # New row max: if pi == k, it's nv; otherwise it's row[k] (unchanged)
    if pi == k:
        new_row_max = nv
    else:
        new_row_max = row[k]
    # Global max: max of all other rows' maxes and new_row_max
    # We need max of other rows. Precompute? For now, compute quickly.
    other_max = 0
    for i, r in enumerate(cur):
        if i != ri:
            m = r[k]  # last element is max since increasing
            if m > other_max:
                other_max = m
    new_max = max(new_row_max, other_max)
    delta = new_max - cur_val
    cand_row = row[:]
    cand_row[pi] = nv
    return cand_row, delta

def _sa_move_row_swap(cur, cur_used, ri, rj, rng, k):
    """Swap the contents of rows ri and rj (as sets of differences).
    This is valid iff the union of differences is unchanged (which it is, since we
    just permute which row owns which differences). The max entry may change."""
    if ri == rj:
        return None, 0
    # Swap the rows
    old_ri = cur[ri][:]
    old_rj = cur[rj][:]
    new_ri = old_rj[:]
    new_rj = old_ri[:]
    # The used set is unchanged (same multiset of differences)
    # Check validity: each row must still be increasing and start at 0 (they are, since we swap whole rows)
    # Compute delta
    old_max = _max_entry(cur)
    new_max = max(max(new_ri), max(new_rj), max(max(r) for i, r in enumerate(cur) if i != ri and i != rj))
    delta = new_max - old_max
    return (new_ri, new_rj), delta

def _sa_move_resolve_row(cur, cur_used, ri, rng, k, cur_val=None):
    """Re-resolve row ri from scratch, avoiding other rows' diffs.
    Returns (new_row, delta) or (None, 0)."""
    other_used = set()
    for i, r in enumerate(cur):
        if i == ri:
            continue
        other_used |= _row_diffs_set(r)
    fresh_used = set(other_used)
    # Try a randomized greedy with a modest target
    old_max = max(cur[ri])
    target = old_max + rng.randint(-10, 10)
    row, _ = _greedy_row_with_update(fresh_used, k, max_val=target, rng=rng)
    if row is None:
        # Try without limit
        row, _ = _greedy_row_with_update(fresh_used, k, rng=rng)
    if row is None:
        return None, 0
    new_diffs = _row_diffs_set(row)
    if new_diffs & other_used:
        return None, 0
    if cur_val is None:
        cur_val = _max_entry(cur)
    other_max = 0
    for i, r in enumerate(cur):
        if i != ri:
            m = r[k]
            if m > other_max:
                other_max = m
    new_max = max(row[k], other_max)
    delta = new_max - cur_val
    return row, delta

def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory starting from initial_rows.
    Returns the best valid rows seen."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    # SA schedule
    T0 = 12.0
    T_end = 0.05
    # We'll run until time_budget expires, with T decaying geometrically
    # Estimate steps per second roughly; we'll just loop and check time
    T = T0
    decay = 0.9995  # per-step decay factor (tuned for a few thousand steps)
    min_T = T_end

    steps = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        steps += 1
        # Choose a move type
        r = rng.random()
        if r < 0.45:
            # Single position move
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k, cur_val)
            if new_row is None:
                T = max(min_T, T * decay)
                continue
            # Accept or reject
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
            # Row swap
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
            # Re-resolve a row
            ri = rng.randrange(n_rows)
            new_row, delta = _sa_move_resolve_row(cur, cur_used, ri, rng, k, cur_val)
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
            # Kick: random restart from best with a small perturbation
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            # Perturb a few positions
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
        # Periodic re-anchoring if we've drifted too far
        if steps % 2000 == 0 and cur_val > best_val + 15:
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val

    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Phase 1: diverse initializations including CSP-based
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(6):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(1000 + s)))
    
    # Try CSP-based initialization
    if time.time() < deadline - 10:
        csp_rows, _ = _ac3_csp_init(n_rows, k, deadline, rng=random.Random(42))
        if csp_rows is not None and _valid(csp_rows, n_rows, k):
            inits.append(csp_rows)
    
    # Pick the best few as starting points for SA
    inits.sort(key=_max_entry)
    top_inits = inits[:4]

    # Phase 2: run multiple SA trajectories with time budget split
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    n_sa = 4
    budget_per_sa = max(5.0, remaining / n_sa - 1.0)

    overall_best = None
    overall_best_val = float('inf')
    for i, init in enumerate(top_inits):
        if time.time() >= deadline - 2:
            break
        b, bv = _sa_run(n_rows, k, init, budget_per_sa, seed=42 + i * 7)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Phase 3: CSP-based repair pass on the best solution
    if overall_best is not None and time.time() < deadline - 5:
        rng_repair = random.Random(777)
        for attempt in range(3):
            if time.time() >= deadline - 3:
                break
            # Try to improve by repairing each row with CSP
            improved = False
            for ri in range(n_rows):
                if time.time() >= deadline - 3:
                    break
                new_row = _csp_repair_row(ri, overall_best, k, deadline, rng_repair)
                if new_row is not None:
                    old_diffs = _row_diffs_set(overall_best[ri])
                    new_diffs = _row_diffs_set(new_row)
                    other_diffs = set()
                    for i, r in enumerate(overall_best):
                        if i != ri:
                            other_diffs |= _row_diffs_set(r)
                    if not (new_diffs & other_diffs):
                        old_max = _max_entry(overall_best)
                        overall_best[ri] = new_row
                        new_max = _max_entry(overall_best)
                        if new_max < old_max:
                            overall_best_val = new_max
                            improved = True
            if not improved:
                break

    # Phase 4: final polish with a short SA from the overall best
    if overall_best is None:
        overall_best = inits[0]
        overall_best_val = _max_entry(overall_best)
    elif time.time() < deadline - 5:
        b, bv = _sa_run(n_rows, k, overall_best, max(3.0, deadline - time.time() - 2), seed=999)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Final validation
    if not _valid(overall_best, n_rows, k):
        overall_best = _build_greedy(n_rows, k, rng=None)

    return overall_best
# EVOLVE-BLOCK-END
