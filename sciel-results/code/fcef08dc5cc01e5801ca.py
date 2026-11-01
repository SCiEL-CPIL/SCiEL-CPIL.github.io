"""Structural rewrite: randomized constructive + DSATUR-style repair."""
import time
import random

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
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    """Set of all pairwise differences in a row (with multiplicity check via list)."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    n = len(row)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _is_row_sidon(row):
    """All pairwise differences within the row must be distinct."""
    d = _row_diffs_list(row)
    return len(d) == len(set(d))

def _check_row_valid(row, k):
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    return True

def _construct_one(n_rows, k, rng, used, bias=1.0, max_try=2000):
    """
    Construct n_rows rows. `used` is the set of diffs already consumed by
    previously-constructed rows (rows are built in order). Returns list of rows
    or None if a row could not be completed within max_try random restarts.
    bias: >1 prefers larger gaps (conflict avoidance); <1 prefers packing.
    """
    rows = []
    for _ in range(n_rows):
        row = None
        for attempt in range(max_try):
            row = [0]
            ok = True
            while len(row) < k + 1:
                # candidate next values: start from last+1 upward
                lo = row[-1] + 1
                # sample a few candidate values with a geometric bias
                best_val = None
                # try a spread of candidates
                candidates = []
                base = lo
                # deterministic small range
                for d in range(1, 12):
                    candidates.append(lo + d - 1)
                # add some larger jumps based on bias
                for d in range(1, 8):
                    candidates.append(int(lo + (d - 1) * bias * 2 + rng.randint(0, 3)))
                candidates = sorted(set(candidates))
                rng.shuffle(candidates)
                # pick the first candidate that doesn't create an internal or global diff conflict
                chosen = None
                for v in candidates:
                    if v <= row[-1]:
                        continue
                    # internal diffs
                    new_internal = [v - a for a in row]
                    # check internal uniqueness (Sidon within row)
                    if len(new_internal) != len(set(new_internal)):
                        continue
                    # check against existing internal diffs of this row
                    existing_internal = set()
                    n = len(row)
                    for i in range(n):
                        for j in range(i + 1, n):
                            existing_internal.add(row[j] - row[i])
                    if set(new_internal) & existing_internal:
                        continue
                    # check against global used
                    if set(new_internal) & used:
                        continue
                    chosen = v
                    break
                if chosen is None:
                    ok = False
                    break
                row.append(chosen)
            if ok:
                # final Sidon check
                if _is_row_sidon(row):
                    break
                else:
                    ok = False
        if not ok or row is None:
            return None
        # register diffs into used
        for d in _row_diffs_list(row):
            used.add(d)
        rows.append(row)
    return rows

def _repair_row(row, k, used, rng, max_try=300):
    """
    Try to rebuild a single row to have a smaller max, given the current
    global `used` set (which should NOT contain this row's diffs).
    Returns a new row with smaller max, or None.
    """
    old_max = row[-1]
    best = None
    for attempt in range(max_try):
        new_row = [0]
        ok = True
        # target: keep max < old_max
        while len(new_row) < k + 1:
            lo = new_row[-1] + 1
            hi = old_max - (k + 1 - len(new_row))  # must leave room
            if hi < lo:
                ok = False
                break
            # candidate range
            candidates = []
            span = hi - lo + 1
            # sample candidates biased toward smaller values (to reduce max)
            for _ in range(20):
                # geometric-ish: prefer near lo
                t = rng.random() ** 2.0
                v = lo + int(t * span)
                candidates.append(v)
            # also include the smallest few
            for d in range(1, min(8, span + 1)):
                candidates.append(lo + d - 1)
            candidates = sorted(set(c for c in candidates if lo <= c <= hi))
            rng.shuffle(candidates)
            chosen = None
            for v in candidates:
                new_internal = [v - a for a in new_row]
                if len(new_internal) != len(set(new_internal)):
                    continue
                existing_internal = set()
                n = len(new_row)
                for i in range(n):
                    for j in range(i + 1, n):
                        existing_internal.add(new_row[j] - new_row[i])
                if set(new_internal) & existing_internal:
                    continue
                if set(new_internal) & used:
                    continue
                chosen = v
                break
            if chosen is None:
                ok = False
                break
            new_row.append(chosen)
        if ok and _is_row_sidon(new_row) and new_row[-1] < old_max:
            if best is None or new_row[-1] < best[-1]:
                best = new_row
                if new_row[-1] == lo:  # can't do better than minimal
                    break
    return best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s (evaluator kills at 150s)
    rng = random.Random(12345)

    best = None
    best_val = None

    # Multi-start: each start builds all rows from scratch with a fresh used set
    start_id = 0
    while time.time() < deadline:
        start_id += 1
        used = set()
        rows = _construct_one(n_rows, k, rng, used, bias=1.0, max_try=1500)
        if rows is None:
            # try a different bias
            used = set()
            rows = _construct_one(n_rows, k, rng, used, bias=1.5, max_try=1500)
        if rows is None:
            # give up this start, try again
            if time.time() > deadline - 5:
                break
            continue

        # Now do DSATUR-style repair: repeatedly rebuild the row with largest max
        # to a smaller value, until no improvement or time runs out.
        cur = [r[:] for r in rows]
        cur_used = set(used)
        cur_val = _max_entry(cur)

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # find row with max entry (tie-break: most constrained = largest max)
            # pick the row index with the largest last element
            target_idx = max(range(n_rows), key=lambda i: cur[i][-1])
            target_row = cur[target_idx]
            # remove its diffs from used
            target_diffs = set(_row_diffs_list(target_row))
            cur_used -= target_diffs
            # try to repair
            new_row = _repair_row(target_row, k, cur_used, rng, max_try=200)
            if new_row is not None:
                # accept: update
                cur[target_idx] = new_row
                cur_used |= set(_row_diffs_list(new_row))
                new_val = _max_entry(cur)
                if new_val < cur_val:
                    cur_val = new_val
                    improved = True
                    if best is None or cur_val < best_val:
                        best_val = cur_val
                        best = [r[:] for r in cur]
                else:
                    # no improvement in max; still keep (may help future repairs)
                    pass
            else:
                # put diffs back
                cur_used |= target_diffs
            # occasionally rotate which row we target to avoid cycling
            if rng.random() < 0.3:
                # try a random row instead
                target_idx = rng.randrange(n_rows)
                target_row = cur[target_idx]
                target_diffs = set(_row_diffs_list(target_row))
                cur_used -= target_diffs
                new_row = _repair_row(target_row, k, cur_used, rng, max_try=150)
                if new_row is not None:
                    cur[target_idx] = new_row
                    cur_used |= set(_row_diffs_list(new_row))
                    new_val = _max_entry(cur)
                    if new_val < cur_val:
                        cur_val = new_val
                        improved = True
                        if best is None or cur_val < best_val:
                            best_val = cur_val
                            best = [r[:] for r in cur]
                else:
                    cur_used |= target_diffs

        # record this start's result
        if best is None or cur_val < best_val:
            if _valid(cur, n_rows, k):
                best_val = cur_val
                best = [r[:] for r in cur]

        # If we already hit the counting bound region, stop early
        if best_val is not None and best_val <= 106:
            break

    # Fallback: if nothing valid was produced, do a pure greedy construction
    if best is None or not _valid(best, n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    # also check internal Sidon
                    if len(new) == len(row):
                        row.append(x)
                        used |= new
            rows.append(row)
        if _valid(rows, n_rows, k):
            best = rows

    return best

# EVOLVE-BLOCK-END
