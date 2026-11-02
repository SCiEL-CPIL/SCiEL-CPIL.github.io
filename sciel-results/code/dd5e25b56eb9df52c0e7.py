"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Compute the set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    """Compute list of all pairwise differences (to check for duplicates)."""
    lst = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            lst.append(row[j] - row[i])
    return lst

def _is_valid_row(row, forbidden, k):
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    diffs = _row_diffs_list(row)
    if len(diffs) != len(set(diffs)):
        return False
    diff_set = set(diffs)
    if diff_set & forbidden:
        return False
    return True

def _random_valid_row(forbidden_diffs, max_limit, k, rng, max_candidates=5):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_diffs,
    with max entry <= max_limit.
    """
    if max_limit is None or max_limit < k:
        return None
    row = [0]
    used_in_row = set()

    for _ in range(k):
        candidates = []
        start = row[-1] + 1
        end = max_limit if max_limit is not None else start + 300

        for x in range(start, end + 1):
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

def _greedy_init(n_rows, k, rng):
    """
    Build initial solution greedily with optional randomization for diversity.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                # occasional small skip for diversity
                if rng.random() < 0.12 and x < 200:
                    skip = rng.randint(1, 3)
                    x2 = x + skip
                    nd2 = {x2 - a for a in row}
                    if not nd2 & used:
                        x = x2
                        new_diffs = nd2
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

def _validate(rows, n_rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diff_list = _row_diffs_list(row)
        if len(diff_list) != len(set(diff_list)):
            return False
        for d in diff_list:
            if d in used:
                return False
            used.add(d)
    return True

def _element_polish(rows, n_rows, k, rng, deadline):
    """
    Element-level simulated annealing: nudge individual entries down (or up)
    to reduce the global max, keeping the row valid and disjoint.
    """
    best = [r[:] for r in rows]
    best_val = _global_max(best)
    cur = [r[:] for r in best]
    cur_val = best_val
    row_diffs = [_row_diffs_set(r) for r in cur]
    all_used = _compute_all_used(cur)

    T = 6.0
    T_min = 0.3
    cool = 0.99995
    no_improve = 0
    max_no_improve = 40000

    while time.time() < deadline and no_improve < max_no_improve:
        ri = rng.randrange(n_rows)
        pi = rng.randrange(1, k + 1)
        old_val = cur[ri][pi]

        # Build candidate new values for position pi
        min_allowed = cur[ri][pi - 1] + 1
        max_allowed = (cur[ri][pi + 1] - 1) if pi < k else cur_val + 20

        # Prefer decreasing moves (to shrink scope)
        candidates = []
        # decreasing
        for v in range(old_val - 1, min_allowed - 1, -1):
            candidates.append(v)
            if len(candidates) >= 12:
                break
        # increasing (small)
        for v in range(old_val + 1, min(old_val + 8, max_allowed + 1)):
            candidates.append(v)

        rng.shuffle(candidates)

        accepted_move = False
        for new_val in candidates:
            if new_val < min_allowed or new_val > max_allowed:
                continue
            new_row = cur[ri][:]
            new_row[pi] = new_val
            # must remain strictly increasing
            ok = True
            for i in range(1, len(new_row)):
                if new_row[i] <= new_row[i - 1]:
                    ok = False
                    break
            if not ok:
                continue
            nd = _row_diffs_list(new_row)
            if len(nd) != len(set(nd)):
                continue
            nds = set(nd)
            # check disjoint from other rows' diffs
            forbidden = all_used - row_diffs[ri]
            if nds & forbidden:
                continue

            # compute new global max
            other_max = _max_except(cur, ri)
            new_max = max(other_max, new_row[-1])
            delta = new_max - cur_val

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.4:
                    accept = True
            else:
                if T > T_min:
                    if rng.random() < math.exp(-delta / T):
                        accept = True

            if accept:
                # apply
                old_d = row_diffs[ri]
                all_used -= old_d
                all_used |= nds
                row_diffs[ri] = nds
                cur[ri] = new_row
                cur_val = new_max
                accepted_move = True

                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    no_improve = 0
                else:
                    no_improve += 1
                break

        if not accepted_move:
            no_improve += 1

        T = max(T_min, T * cool)

        if no_improve > 15000:
            cur = [r[:] for r in best]
            cur_val = best_val
            row_diffs = [_row_diffs_set(r) for r in cur]
            all_used = _compute_all_used(cur)
            T = 6.0
            no_improve = 0

    return best, best_val

def _wl_row_sa(rows, n_rows, k, rng, deadline):
    """
    Wang-Landau-inspired row-level SA.
    Maintain a visit-count (density-of-states estimate) over row-max 'energy' bins.
    Bias sampling of new rows toward low-energy (small-max) bins to flatten the
    landscape and target the minimal-scope region.
    """
    cur = [r[:] for r in rows]
    cur_val = _global_max(cur)
    row_diffs = [_row_diffs_set(r) for r in cur]
    all_used = _compute_all_used(cur)

    best = [r[:] for r in cur]
    best_val = cur_val

    # Wang-Landau: energy = row max, bin width 1, range [k, cur_val+30]
    e_min = k
    e_max = cur_val + 30
    f = {}  # visit count per energy bin
    for e in range(e_min, e_max + 1):
        f[e] = 1.0
    f_target = 100.0  # stop WL refinement when all bins reach target
    f_factor = 1.1

    # SA temperature for accepting worse global max
    T = 5.0
    T_min = 0.01
    cool = 0.995
    no_improve = 0
    max_no_improve = 30000

    while time.time() < deadline and no_improve < max_no_improve:
        # Preferentially target rows with high max
        if rng.random() < 0.6:
            hi = [i for i in range(n_rows) if cur[i][-1] >= cur_val - 5]
            ri = rng.choice(hi) if hi else rng.randrange(n_rows)
        else:
            ri = rng.randrange(n_rows)

        forbidden = all_used - row_diffs[ri]

        # Wang-Landau-guided target: sample an energy bin weighted by 1/f
        # and attempt to build a row with max near that bin (low bins preferred)
        # Build candidate target maxes weighted toward low energy
        cand_maxes = []
        cand_w = []
        for e in range(e_min, min(e_max, cur_val + 10) + 1):
            w = 1.0 / f[e]
            cand_maxes.append(e)
            cand_w.append(w)

        total_w = sum(cand_w)
        r = rng.random() * total_w
        acc = 0.0
        target_max = cand_maxes[-1]
        for e, w in zip(cand_maxes, cand_w):
            acc += w
            if r <= acc:
                target_max = e
                break

        new_row = _random_valid_row(forbidden, target_max, k, rng, max_candidates=8)

        if new_row is None:
            new_row = _random_valid_row(forbidden, cur_val, k, rng, max_candidates=8)

        if new_row is None:
            new_row = _random_valid_row(forbidden, cur_val + 8, k, rng, max_candidates=8)

        if new_row is None:
            no_improve += 1
            T = max(T_min, T * cool)
            continue

        # Update Wang-Landau visit count for the achieved energy
        e_achieved = new_row[-1]
        if e_achieved not in f:
            f[e_achieved] = 1.0
        f[e_achieved] += 1.0
        # If all bins reached target, refine f (multiplicative update)
        if min(f.values()) >= f_target:
            for key in f:
                f[key] = 1.0
            f_target *= f_factor

        other_max = _max_except(cur, ri)
        new_max = max(other_max, new_row[-1])
        delta = new_max - cur_val

        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if rng.random() < 0.45:
                accept = True
        else:
            if T > T_min:
                if rng.random() < math.exp(-delta / T):
                    accept = True

        if accept:
            old_d = row_diffs[ri]
            all_used -= old_d
            all_used |= _row_diffs_set(new_row)
            row_diffs[ri] = all_used - (all_used - _row_diffs_set(new_row))
            row_diffs[ri] = _row_diffs_set(new_row)
            cur[ri] = new_row
            cur_val = new_max

            if cur_val < best_val:
                best_val = cur_val
                best = [r[:] for r in cur]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        T = max(T_min, T * cool)

        if no_improve > 12000:
            cur = [r[:] for r in best]
            cur_val = best_val
            row_diffs = [_row_diffs_set(r) for r in cur]
            all_used = _compute_all_used(cur)
            T = 5.0
            no_improve = 0

    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Multi-start: several diverse greedy inits, keep best
    best_rows = None
    best_val = 10**9
    for seed in range(4):
        rng0 = random.Random(1000 + seed)
        rows = _greedy_init(n_rows, k, rng0)
        v = _global_max(rows)
        if v < best_val:
            best_val = v
            best_rows = [r[:] for r in rows]

    if best_rows is None:
        best_rows = _greedy_init(n_rows, k, random.Random(42))
        best_val = _global_max(best_rows)

    # Wang-Landau row-level SA on the best init
    rng = random.Random(777)
    best_rows, best_val = _wl_row_sa(best_rows, n_rows, k, rng, deadline)

    # Element-level polish
    if time.time() < deadline - 5:
        best_rows, best_val = _element_polish(best_rows, n_rows, k, rng, deadline)

    # Final validation
    if not _validate(best_rows, n_rows, k):
        best_rows = _greedy_init(n_rows, k, random.Random(42))

    return best_rows

# EVOLVE-BLOCK-END
