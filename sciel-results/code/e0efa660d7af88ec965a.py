import time
import random
import sys
import math

sys.setrecursionlimit(100000)


def construct(n_rows, k):
    start = time.time()
    deadline = start + 145  # hard wall; evaluator kills at 150s

    # 1) Fast multi-start greedy baseline
    best = _greedy_multi(n_rows, k, budget=6.0)
    if not best:
        return []
    best_scope = max(max(r) for r in best)

    # 2) Bitset local search on the difference set (WalkSAT analogue)
    t2 = deadline - 12
    improved = _bitset_local_search(best, n_rows, k, best_scope, t2)
    if improved:
        s = max(max(r) for r in improved)
        if s < best_scope:
            best_scope = s
            best = improved

    # 3) Row-level element SA (kept as a polish)
    if time.time() < deadline - 6:
        improved = _element_sa(best, n_rows, k, deadline - 4)
        if improved:
            s = max(max(r) for r in improved)
            if s < best_scope:
                best_scope = s
                best = improved

    # 4) Final deterministic tightening
    if time.time() < deadline - 1.5:
        tightened = _tighten(best, n_rows, k, deadline - 0.5)
        if tightened:
            s = max(max(r) for r in tightened)
            if s < best_scope:
                best = tightened

    return best


# ---------- helpers ----------

def _row_diff_set(row):
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _all_diff_set(rows):
    s = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                s.add(r[j] - ri)
    return s


def _diff_bitset(rows):
    """Return an int whose bit d is set iff d appears as a positive diff somewhere."""
    m = 0
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                m |= (1 << (r[j] - ri))
    return m


def _bitset_to_set(m):
    s = set()
    d = 0
    while m:
        if m & 1:
            s.add(d)
        m >>= 1
        d += 1
    return s


def _valid_rows_from_diffs(rows, k):
    """Check each row internally, and global uniqueness across rows."""
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in used:
                    return False
                used.add(d)
        for i in range(1, len(r)):
            if r[i] <= r[i - 1]:
                return False
    return True


# ---------- greedy ----------

def _greedy_one_row(used, k):
    row = [0]
    while len(row) < k + 1:
        last = row[-1]
        found = False
        # scan upward
        for x in range(last + 1, last + 4000):
            ok = True
            for a in row:
                if (1 << (x - a)) & used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used |= (1 << (x - a))
                found = True
                break
        if not found:
            return None
    return row


def _greedy_multi(n_rows, k, budget):
    """Multi-start greedy with varied random perturbations of the row construction."""
    t_end = time.time() + budget
    best_rows = None
    best_scope = float('inf')

    # Deterministic first
    used = 0
    rows = []
    for _ in range(n_rows):
        r = _greedy_one_row(used, k)
        if r is None:
            rows = None
            break
        rows.append(r)
    if rows:
        best_rows = rows
        best_scope = max(max(r) for r in rows)

    # Randomized greedy: shuffle candidate ordering slightly
    attempts = 0
    while time.time() < t_end and attempts < 30:
        attempts += 1
        used = 0
        rows = []
        ok = True
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                # try a small random window then expand
                window = random.randint(15, 60)
                found = False
                for _w in range(2):
                    lo = last + 1
                    hi = last + window
                    order = list(range(lo, hi + 1))
                    random.shuffle(order)
                    for x in order:
                        if (1 << x) & used:
                            continue
                        ok2 = True
                        for a in row:
                            if (1 << (x - a)) & used:
                                ok2 = False
                                break
                        if ok2:
                            row.append(x)
                            for a in row[:-1]:
                                used |= (1 << (x - a))
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # expand
                    for x in range(last + 1, last + 2500):
                        ok2 = True
                        for a in row:
                            if (1 << (x - a)) & used:
                                ok2 = False
                                break
                        if ok2:
                            row.append(x)
                            for a in row[:-1]:
                                used |= (1 << (x - a))
                            found = True
                            break
                if not found:
                    ok = False
                    break
            if not ok:
                break
            rows.append(row)
        if ok and rows:
            s = max(max(r) for r in rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
    return best_rows


# ---------- Bitset WalkSAT on the difference set ----------
# Variables: d in [1..max_scope] -> whether d is "used"
# Hard constraints: each row must be able to be reconstructed so that the set of
#   its k+1 choose 2 differences is a subset of used, with no overlap across rows.
# We measure conflict = number of rows that cannot be reconstructed.

def _try_build_row(used_bit, k, max_val):
    """Greedy: build a row of k+1 entries from [0..max_val] using only diffs in used_bit.
    Return row or None."""
    row = [0]
    while len(row) < k + 1:
        last = row[-1]
        found = False
        for x in range(last + 1, max_val + 1):
            ok = True
            for a in row:
                if not ((1 << (x - a)) & used_bit):
                    ok = False
                    break
            if ok:
                row.append(x)
                found = True
                break
        if not found:
            return None
    return row


def _rebuild_rows_from_bitset(used_bit, k, max_val):
    """Try to pack n_rows=7 rows (or whatever we need) from the diff set.
    We attempt up to a fixed number of times, rebuilding one row at a time,
    removing its diffs from the available pool."""
    # We don't know n_rows here; caller passes it.
    return None


def _bitset_local_search(initial_rows, n_rows, k, initial_scope, deadline):
    """WalkSAT over the difference-set bitset.

    State: used_bit (int) — the set of differences currently 'allowed/used'.
    Objective: be able to greedily reconstruct n_rows valid rows from this set,
    with max entry <= scope_target. We minimize the number of rows that fail to
    rebuild (conflict count), then the max entry.
    """
    rows = [r[:] for r in initial_rows]
    used_bit = _diff_bitset(rows)
    scope_target = max(max(r) for r in rows)

    # Precompute per-row diff bitset
    row_bits = [_diff_bitset([r]) for r in rows]

    # Conflict: for each row, can we rebuild it from (used_bit & row_bits) ?
    # Actually if we keep the row's own diffs, it's trivially rebuildable.
    # Real conflict appears when we REMOVE a diff that a row needs.
    # So we track which diffs are "required" by how many rows.
    req_count = [0] * (scope_target + 1)
    for rb in row_bits:
        b = rb
        d = 0
        while b:
            if b & 1:
                req_count[d] += 1
            b >>= 1
            d += 1

    def row_conflict(idx):
        # A row is in conflict if any of its required diffs is currently absent
        rb = row_bits[idx]
        b = rb
        d = 0
        while b:
            if b & 1:
                if not (used_bit >> d) & 1:
                    return True
            b >>= 1
            d += 1
        return False

    def total_conflict():
        c = 0
        for i in range(n_rows):
            if row_conflict(i):
                c += 1
        return c

    best_conf = total_conflict()
    best_rows_snapshot = [r[:] for r in rows]
    best_bit = used_bit
    best_scope = scope_target

    T = 8.0
    Tmin = 0.05
    cool = 0.99999
    no_improve = 0

    # Max diff value we allow the bitset to cover (we can grow it)
    max_diff = scope_target + 40

    while time.time() < deadline - 2:
        # Pick a row (biased toward conflicted) and a diff in it
        conflicted = [i for i in range(n_rows) if row_conflict(i)]
        if conflicted:
            ri = random.choice(conflicted)
        else:
            ri = random.randrange(n_rows)

        # Choose a diff to flip: prefer one that reduces conflict
        rb = row_bits[ri]
        diff_list = []
        b = rb
        d = 0
        while b:
            if b & 1:
                diff_list.append(d)
            b >>= 1
            d += 1

        if not diff_list:
            break

        # For each candidate diff, evaluate delta conflict
        # Flipping a diff means: if it's currently in used_bit, remove it; if not, add it.
        # But "adding" a diff not currently used doesn't help rebuild (rebuild needs the
        # diff to BE in the bitset). So the useful flip is: REMOVE a diff that is
        # over-represented (req_count>1), or ADD a diff that a row needs.
        # In our representation, every row's diffs are a subset of used_bit (invariant).
        # Flipping = removing. After removal, rows that need it conflict.
        # To repair, we ADD a different diff (one not used, but that lets the row
        # be rebuilt with a different value). This is the WalkSAT move.

        # Choose a diff to remove
        d_remove = random.choice(diff_list)
        # Choose a diff to add (from a small random window near the row's values)
        row = rows[ri]
        # candidate add diffs: differences between new potential values
        # We generate a candidate new value for some position in row ri
        pos = random.randrange(1, k + 1)
        lo = row[pos - 1] + 1
        hi = (row[pos + 1] - 1) if pos < k else scope_target
        if hi < lo:
            hi = lo
        cand_vals = []
        for _ in range(12):
            v = random.randint(lo, min(hi, scope_target + 20))
            cand_vals.append(v)
        # Also try smaller values to reduce scope
        for v in range(lo, min(lo + 25, row[pos] + 1)):
            cand_vals.append(v)

        moved = False
        for new_val in cand_vals:
            if new_val <= row[pos - 1]:
                continue
            if pos < k and new_val >= row[pos + 1]:
                continue

            # Compute new row
            old_row = rows[ri]
            new_row = old_row[:]
            new_row[pos] = new_val

            # Internal validity of new_row
            old_diffs = set()
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    old_diffs.add(old_row[j] - old_row[i])

            new_diffs = set()
            ok = True
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = new_row[j] - new_row[i]
                    if d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
            if not ok:
                continue

            # Cross-row conflict: new_diffs \ old_diffs must not intersect
            # any other row's diffs
            other_conflict = False
            for oi in range(n_rows):
                if oi == ri:
                    continue
                od = _row_diff_set(rows[oi])
                for nd in new_diffs:
                    if nd in od:
                        other_conflict = True
                        break
                if other_conflict:
                    break
            if other_conflict:
                continue

            # Move is valid. Compute new scope.
            new_scope = max(max(r) for i, r in enumerate(rows) if i != ri for _ in [0]) if n_rows > 1 else max(new_row)
            # recompute properly
            maxes = []
            for i, r in enumerate(rows):
                if i == ri:
                    maxes.append(max(new_row))
                else:
                    maxes.append(max(r))
            new_scope = max(maxes)

            delta_scope = new_scope - best_scope if new_scope > best_scope else (new_scope - scope_target)
            # Acceptance: prefer lower scope; allow small increases for exploration
            if new_scope <= scope_target:
                accept = True
            else:
                accept = random.random() < math.exp(-(new_scope - scope_target) / max(T, 0.05))

            if accept:
                # Commit
                rows[ri] = new_row
                row_bits[ri] = _diff_bitset([new_row])
                used_bit = 0
                for i in range(n_rows):
                    used_bit |= row_bits[i]
                scope_target = new_scope
                # Update req_count (recompute; cheap enough for small scope)
                req_count = [0] * (scope_target + 2)
                for rb in row_bits:
                    b = rb
                    d = 0
                    while b:
                        if b & 1:
                            if d < len(req_count):
                                req_count[d] += 1
                            else:
                                req_count.extend([0] * (d + 2 - len(req_count)))
                                req_count[d] += 1
                        b >>= 1
                        d += 1

                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows_snapshot = [r[:] for r in rows]
                    best_bit = used_bit
                    no_improve = 0
                else:
                    no_improve += 1
                moved = True
                break

        if not moved:
            no_improve += 1

        T = max(Tmin, T * cool)

        if no_improve > 30000:
            # restart with current best
            rows = [r[:] for r in best_rows_snapshot]
            row_bits = [_diff_bitset([r]) for r in rows]
            used_bit = best_bit
            scope_target = best_scope
            no_improve = 0
            T = max(T, 5.0)

    # Final validation
    if _valid_rows_from_diffs(best_rows_snapshot, k):
        return best_rows_snapshot
    return None


# ---------- Element SA (kept, optimized) ----------

def _element_sa(rows, n_rows, k, deadline):
    rows = [r[:] for r in rows]
    used = _all_diff_set(rows)
    row_diffs = [_row_diff_set(r) for r in rows]
    row_maxes = [max(r) for r in rows]
    cur_scope = max(row_maxes)
    best = [r[:] for r in rows]
    best_scope = cur_scope

    T = 120.0
    Tmin = 0.2
    cool = 0.99998

    no_improve = 0
    max_no_improve = 60000

    while time.time() < deadline - 2 and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        row = rows[ri]
        old_val = row[pi]

        # Candidates
        cands = []
        lo = row[pi - 1] + 1
        hi = (row[pi + 1] - 1) if pi < k else cur_scope + 30
        # bias toward smaller
        for d in range(1, min(30, old_val - lo + 1)):
            cands.append(old_val - d)
        for d in range(1, min(10, hi - old_val + 1)):
            cands.append(old_val + d)
        if random.random() < 0.1:
            cands.append(random.randint(lo, hi))
        random.shuffle(cands)

        moved = False
        for nv in cands:
            if nv <= row[pi - 1]:
                continue
            if pi < k and nv >= row[pi + 1]:
                continue
            # internal + cross check
            nd = set()
            ok = True
            for i in range(k + 1):
                if i == pi:
                    continue
                d = abs(nv - row[i])
                if d in nd:
                    ok = False
                    break
                nd.add(d)
            if not ok:
                continue
            # cross: nd - old_row_diffs must not be in used - old_row_diffs
            old_rd = row_diffs[ri]
            conflict = False
            forbidden = used - old_rd
            for d in nd:
                if d in forbidden:
                    conflict = True
                    break
            if conflict:
                continue

            # new scope
            if pi == k:
                new_max = nv
            else:
                new_max = row[k]
            old_rm = row_maxes[ri]
            new_scope = max(max(row_maxes) if False else (new_max if new_max > max([m for i, m in enumerate(row_maxes) if i != ri]) else max([m for i, m in enumerate(row_maxes) if i != ri])))

            delta = new_scope - cur_scope
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.2
            else:
                accept = random.random() < math.exp(-delta / max(T, 0.1))

            if accept:
                used -= old_rd
                used |= nd
                row[pi] = nv
                row_diffs[ri] = nd
                row_maxes[ri] = new_max
                cur_scope = new_scope
                if cur_scope < best_scope:
                    best_scope = cur_scope
                    best = [r[:] for r in rows]
                    no_improve = 0
                else:
                    no_improve += 1
                moved = True
                break

        if not moved:
            no_improve += 1
        T = max(Tmin, T * cool)

    if _valid_rows_from_diffs(best, k):
        return best
    return None


# ---------- Deterministic tightening ----------

def _tighten(rows, n_rows, k, deadline):
    rows = [r[:] for r in rows]
    used = _all_diff_set(rows)
    progress = True
    while progress and time.time() < deadline - 0.5:
        progress = False
        for ri in range(n_rows):
            row = rows[ri]
            for pi in range(k, 0, -1):
                old_val = row[pi]
                lo = row[pi - 1] + 1
                hi = old_val - 1
                if pi < k:
                    hi = min(hi, row[pi + 1] - 1)
                if hi < lo:
                    continue
                old_d = set()
                for i in range(k + 1):
                    if i == pi:
                        continue
                    old_d.add(abs(old_val - row[i]))
                forbidden = used - old_d
                found = None
                nd = None
                for nv in range(lo, hi + 1):
                    cand = set()
                    ok = True
                    for i in range(k + 1):
                        if i == pi:
                            continue
                        d = abs(nv - row[i])
                        if d in cand or d in forbidden:
                            ok = False
                            break
                        cand.add(d)
                    if ok:
                        found = nv
                        nd = cand
                        break
                if found is not None:
                    used -= old_d
                    used |= nd
                    row[pi] = found
                    progress = True
    if _valid_rows_from_diffs(rows, k):
        return rows
    return None


# Keep original helper names that the seed used (in case any are referenced)
def _row_diffs(row):
    return _row_diff_set(row)


def _compute_all_diffs(rows):
    return _all_diff_set(rows)


def _greedy_construct(n_rows, k):
    used = 0
    rows = []
    for _ in range(n_rows):
        r = _greedy_one_row(used, k)
        if r is None:
            return None
        rows.append(r)
    return rows


def _greedy_construct_fast(n_rows, k):
    return _greedy_multi(n_rows, k, budget=3.0)


def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    row = rows[r_idx]
    used = _all_diff_set(rows)
    old_d = set()
    for i in range(k + 1):
        if i == pos:
            continue
        old_d.add(abs(row[pos] - row[i]))
    forbidden = used - old_d
    lo = row[pos - 1] + 1
    hi = (row[pos + 1] - 1) if pos < k else max(max(r) for r in rows) + 20
    cands = []
    for nv in range(lo, hi + 1):
        if nv == row[pos]:
            continue
        cand = set()
        ok = True
        for i in range(k + 1):
            if i == pos:
                continue
            d = abs(nv - row[i])
            if d in cand or d in forbidden:
                ok = False
                break
            cand.add(d)
        if ok:
            cands.append(nv)
    if not cands:
        return None
    return random.choice(cands)


def _build_static_diffs(row, k):
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                s.add(row[j] - row[i])
        static[pi] = s
    return static


def _compute_new_diffs_fast(row, pi, new_val, static_diffs_pi):
    n = len(row)
    new_diffs = set(static_diffs_pi)
    valid = True
    for i in range(n):
        if i == pi:
            continue
        d = abs(new_val - row[i])
        if d in new_diffs:
            valid = False
            break
        new_diffs.add(d)
    return new_diffs, valid


def _simulated_annealing(rows, n_rows, k, deadline):
    return _element_sa(rows, n_rows, k, deadline)


def _local_search_minimize(rows, n_rows, k, deadline):
    r = _tighten(rows, n_rows, k, deadline)
    return r if r else [x[:] for x in rows]
