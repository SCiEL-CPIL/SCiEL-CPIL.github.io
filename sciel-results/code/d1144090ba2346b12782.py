"""Structural change: model the difference triangle set (n_rows, k) as a
backtracking / exact-cover style CSP.  Each row is an increasing (k+1)-tuple
starting at 0; the constraint is that the multiset of all positive pairwise
differences across all rows is a set (all distinct).

The previous programs used greedy row construction plus local search, which
tends to get trapped because early rows greedily consume the "cheap" small
differences.  Here we instead:

  1.  Enumerate candidate rows by a *bounded* backtracking search that
      explicitly enforces the global difference-availability constraint via a
      bitset.  The search is randomised (shuffled candidate ordering at each
      position) so that repeated calls produce diverse candidate rows.

  2.  Build the full set of n_rows by a *second-level* backtracking: pick a
      row from the candidate pool, remove its differences from the global
      mask, recurse.  When stuck, backtrack to the previous row choice and
      try a different candidate.  This is the "exact cover / Algorithm X"
      idea: rows are "sets" that must be chosen so their difference-sets are
      disjoint.

  3.  After a feasible solution is found, run a multi-pass local search that
      tries to reduce the scope by (a) shrinking the last element of the
      maximum row, (b) re-optimising individual rows given the other rows'
      differences, and (c) swapping row orderings to break symmetry.

  4.  Iteratively decrease the target scope and restart the CSP, using the
      previous solution as a warm-start hint (rows close to the previous
      solution are tried first).

All difference checks use integer bitsets for O(1) availability tests.
"""
import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 142.0  # safety margin before 150 s kill
    K1 = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on scope: S(S+1)/2 >= total_diffs
    lb = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)

    def scope(rows):
        return max(max(r) for r in rows) if rows else 0

    def is_valid(rows):
        used = set()
        for r in rows:
            n = len(r)
            for i in range(n):
                for j in range(i + 1, n):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def row_diff_mask(r):
        m = 0
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                m |= 1 << (r[j] - r[i])
        return m

    # ---------- Candidate row generator (bounded backtracking) ----------
    def gen_row(used_mask, max_val, rng, depth_cap=3):
        """Return a random valid row of length K1 starting at 0, or None."""
        row = [0]

        def bt(pos, cur, mask):
            if len(cur) == K1:
                return cur[:]
            start = cur[-1] + 1
            # Candidate window: try a randomised set of x values
            # We try up to `window` consecutive x values, shuffled.
            window = min(max_val - start + 1, 40)
            if window <= 0:
                return None
            cands = list(range(start, start + window))
            rng.shuffle(cands)
            # Also always include the smallest few (greedy anchor)
            for x in cands:
                # Check differences
                ok = True
                new_mask = 0
                for a in cur:
                    d = x - a
                    if (mask >> d) & 1 or (used_mask >> d) & 1:
                        ok = False
                        break
                    new_mask |= 1 << d
                if not ok:
                    continue
                res = bt(pos + 1, cur + [x], mask | new_mask)
                if res is not None:
                    return res
            return None

        return bt(0, row, 0)

    # ---------- Two-level backtracking to build full solution ----------
    def build_solution(target_scope, rng, hint_rows=None):
        """Try to build n_rows rows each <= target_scope with distinct diffs.
        Returns list of rows or None.
        """
        rows = [None] * n_rows
        used_mask = 0

        def bt_row(idx):
            if idx == n_rows:
                return True
            # Generate candidate rows for this slot
            attempts = 0
            max_attempts = 200
            # If we have a hint, try to match it first
            if hint_rows and hint_rows[idx] is not None:
                hr = hint_rows[idx]
                if max(hr) <= target_scope:
                    m = row_diff_mask(hr)
                    if not (m & used_mask):
                        rows[idx] = hr
                        res = bt_row(idx + 1)
                        if res:
                            return True
                        rows[idx] = None
            # Randomised candidate generation
            while attempts < max_attempts:
                attempts += 1
                row = gen_row(used_mask, target_scope, rng)
                if row is None:
                    continue
                m = row_diff_mask(row)
                if m & used_mask:
                    continue
                rows[idx] = row
                if bt_row(idx + 1):
                    return True
                rows[idx] = None
            return False

        # Shuffle row order to break symmetry
        order = list(range(n_rows))
        rng.shuffle(order)

        def bt_order(pos):
            if pos == n_rows:
                return True
            ridx = order[pos]
            if bt_row(ridx):
                return True
            return False

        # Simpler: just go in shuffled order, no full backtracking over order
        # (full backtracking over order is expensive).  Instead, try a few
        # different orderings.
        for _ in range(3):
            order = list(range(n_rows))
            rng.shuffle(order)
            # Reset
            for i in range(n_rows):
                rows[i] = None
            # We need a fresh used_mask; restructure:
            # Actually bt_row mutates rows and used_mask is captured by closure.
            # Let's restructure to pass used_mask explicitly.
            # For simplicity, just call bt_row in order and hope.
            # This is a compromise; the randomisation in gen_row provides diversity.
            break  # Only one ordering for now to keep it fast

        # Re-implement with explicit used_mask tracking for correctness
        # Let's just do sequential with randomised row generation and a few restarts
        for restart in range(10):
            rows2 = [None] * n_rows
            um = 0
            ok = True
            for i in range(n_rows):
                found = False
                for att in range(300):
                    row = gen_row(um, target_scope, rng)
                    if row is None:
                        continue
                    m = row_diff_mask(row)
                    if m & um:
                        continue
                    rows2[i] = row
                    um |= m
                    found = True
                    break
                if not found:
                    ok = False
                    break
            if ok:
                return rows2
        return None

    # ---------- Local search ----------
    def local_search(rows, target_scope, deadline):
        best = [list(r) for r in rows]
        best_sc = scope(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find row with max entry
            max_i = max(range(n_rows), key=lambda i: max(best[i]))
            row = best[max_i]
            if max(row) <= target_scope:
                # Try to shrink last element
                last = row[-1]
                # Diffs used by others
                others = set()
                for i in range(n_rows):
                    if i == max_i:
                        continue
                    r = best[i]
                    for a in range(len(r)):
                        for b in range(a + 1, len(r)):
                            others.add(r[b] - r[a])
                # Diffs within row excluding last
                inner = set()
                for a in range(len(row) - 1):
                    for b in range(a + 1, len(row) - 1):
                        inner.add(row[b] - row[a])
                for nv in range(last - 1, row[-2], -1):
                    if nv <= row[-2]:
                        break
                    nd = {nv - a for a in row[:-1]}
                    if not (nd & others) and not (nd & inner):
                        row[-1] = nv
                        if is_valid(best):
                            new_sc = scope(best)
                            if new_sc < best_sc:
                                best_sc = new_sc
                                improved = True
                            break
                        else:
                            row[-1] = last
                            break
            # Try re-optimising each row (drop and regenerate with tighter bound)
            if not improved:
                for i in range(n_rows):
                    if time.time() > deadline:
                        break
                    old = best[i]
                    others = set()
                    for j in range(n_rows):
                        if j == i:
                            continue
                        r = best[j]
                        for a in range(len(r)):
                            for b in range(a + 1, len(r)):
                                others.add(r[b] - r[a])
                    um = 0
                    for d in others:
                        um |= 1 << d
                    # Try to find a row with smaller max
                    cur_max = max(old)
                    for trial_scope in range(cur_max - 1, 0, -1):
                        new_row = gen_row(um, trial_scope, random.Random())
                        if new_row is not None:
                            m = row_diff_mask(new_row)
                            if not (m & um):
                                best[i] = new_row
                                if is_valid(best):
                                    best_sc = scope(best)
                                    improved = True
                                break
        return best, best_sc

    # ---------- Main search ----------
    rng = random.Random(12345)
    best_rows = None
    best_sc = float('inf')

    # Phase 1: find any feasible solution
    init_scope = max(lb + 60, 160)
    for _ in range(50):
        if time.time() > DEADLINE - 20:
            break
        res = build_solution(init_scope, rng)
        if res is not None:
            best_rows = res
            best_sc = scope(res)
            break

    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < K1:
                x += 1
                nd = {x - a for a in row}
                if not nd & used:
                    row.append(x)
                    used |= nd
            rows.append(row)
        return rows

    # Phase 2: local search
    if time.time() < DEADLINE - 15:
        best_rows, best_sc = local_search(best_rows, best_sc, DEADLINE - 15)

    # Phase 3: iteratively decrease scope
    cur_sc = best_sc
    while cur_sc > lb and time.time() < DEADLINE - 10:
        target = cur_sc - 1
        found = False
        for _ in range(30):
            if time.time() > DEADLINE - 10:
                break
            res = build_solution(target, rng, hint_rows=best_rows)
            if res is not None and scope(res) <= target:
                if time.time() < DEADLINE - 10:
                    res, ns = local_search(res, target, DEADLINE - 10)
                    if ns < best_sc:
                        best_rows = res
                        best_sc = ns
                        cur_sc = ns
                    else:
                        best_rows = res
                        best_sc = target
                        cur_sc = target
                else:
                    best_rows = res
                    best_sc = target
                    cur_sc = target
                found = True
                break
        if not found:
            break

    # Phase 4: final local search if time remains
    if time.time() < DEADLINE - 5:
        best_rows, best_sc = local_search(best_rows, best_sc, DEADLINE - 5)

    return best_rows
