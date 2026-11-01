import time
import random

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 115.0

    total_diffs_per_row = k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)
    rng = random.Random(42)

    # Constraint-propagation guided local search
    # Key insight: track which differences are "conflicted" (appear in multiple rows)
    # and preferentially move elements that create/resolve conflicts.
    # Use incremental validation: when moving element at pos in row ri,
    # only need to check diffs involving that element against all other rows' diffs.

    def build_diff_index(rows):
        """Build a dict: diff -> list of (row_idx, i, j) pairs that produce it."""
        idx = {}
        for ri, row in enumerate(rows):
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d not in idx:
                        idx[d] = []
                    idx[d].append((ri, i, j))
        return idx

    def row_diffs_set(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def fast_validate_move(rows, ri, pos, new_val, diff_index):
        """
        Check if setting rows[ri][pos] = new_val keeps validity.
        Only checks differences involving position pos in row ri.
        Returns (valid, new_scope) or (False, None).
        """
        row = rows[ri]
        old_val = row[pos]
        if new_val == old_val:
            return True, get_scope(rows)
        # Check ordering
        if pos > 0 and new_val <= row[pos - 1]:
            return False, None
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False, None

        # Compute old diffs involving pos
        old_diffs = set()
        for j in range(pos):
            old_diffs.add(row[pos] - row[j])
        for j in range(pos + 1, len(row)):
            old_diffs.add(row[j] - row[pos])

        # Compute new diffs involving pos
        new_diffs = set()
        for j in range(pos):
            new_diffs.add(new_val - row[j])
        for j in range(pos + 1, len(row)):
            new_diffs.add(row[j] - new_val)

        # Check: new_diffs must not conflict with any diff in other rows
        # that was NOT also in old_diffs (those were already accounted for)
        # Actually simpler: new_diffs must not appear in any other row's diff set,
        # AND must not appear twice within this row (but since pos is fixed,
        # the diffs involving pos are unique by construction if row is strictly increasing).
        # Wait - within the same row, two different pairs could give same diff.
        # But we only changed one element, so we need to check:
        # 1. new_diffs don't overlap with diffs from OTHER rows
        # 2. new_diffs don't overlap with each other (shouldn't happen for strict increase)
        # 3. new_diffs don't overlap with old diffs from OTHER positions in this row
        #    that don't involve pos (those are unchanged)

        # Build set of all diffs in other rows
        other_diffs = set()
        for oi in range(n_rows):
            if oi == ri:
                continue
            orow = rows[oi]
            for i in range(len(orow)):
                for j in range(i + 1, len(orow)):
                    other_diffs.add(orow[j] - orow[i])

        # Diffs in row ri that don't involve pos (unchanged)
        unchanged_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            for j in range(i + 1, len(row)):
                if j == pos:
                    continue
                unchanged_diffs.add(row[j] - row[i])

        # Check conflicts
        for d in new_diffs:
            if d in other_diffs:
                return False, None
            if d in unchanged_diffs:
                return False, None
        # Also check new_diffs internal (should be fine for strict increase)
        if len(new_diffs) != sum(1 for _ in new_diffs):
            return False, None

        # Compute new scope
        new_scope = 0
        for r in rows:
            new_scope = max(new_scope, max(r))
        # Account for the changed value
        if new_val > new_scope:
            new_scope = new_val
        # Actually just compute properly
        new_scope = max(max(r) for r in rows)
        return True, new_scope

    # Better approach: precompute per-row diff sets for O(1) conflict checks
    def build_row_diff_sets(rows):
        return [row_diffs_set(r) for r in rows]

    def check_move_conflict(rows, row_diff_sets, ri, pos, new_val):
        """
        O(k) check: does setting rows[ri][pos]=new_val cause any diff conflict?
        """
        row = rows[ri]
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False

        new_diffs = set()
        for j in range(pos):
            new_diffs.add(new_val - row[j])
        for j in range(pos + 1, len(row)):
            new_diffs.add(row[j] - new_val)

        # Check against other rows' diff sets
        for oi in range(n_rows):
            if oi == ri:
                continue
            if new_diffs & row_diff_sets[oi]:
                return False

        # Check within same row: new_diffs vs unchanged diffs
        # Unchanged diffs are those not involving pos
        for i in range(len(row)):
            if i == pos:
                continue
            for j in range(i + 1, len(row)):
                if j == pos:
                    continue
                d = row[j] - row[i]
                if d in new_diffs:
                    return False

        return True

    def update_row_diff_set(row):
        return row_diffs_set(row)

    def constraint_guided_search(rows, budget):
        """
        Structural change: constraint-propagation guided descent.
        Instead of random walks, we:
        1. Identify the 'bottleneck' element (the one whose value most limits scope reduction)
        2. Find minimal valid value for that element using constraint propagation
        3. Move it there, then re-propagate
        """
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        row_diff_sets = build_row_diff_sets(cur)
        improved = False

        while time.time() - start < budget:
            # Find row with max value
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]

            # Try to reduce each element from the last (highest) position
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                lo = row[pos - 1] + 1
                hi = row[pos + 1] - 1 if pos < len(row) - 1 else old_val - 1

                if hi <= lo:
                    continue

                # Try values from lo upward (smallest valid = best for scope)
                # but also try some random values for diversity
                candidates = []
                # Greedy: try smallest values first
                for v in range(lo, min(lo + 30, hi + 1)):
                    candidates.append(v)
                # Random samples
                for _ in range(15):
                    candidates.append(rng.randint(lo, hi))

                found = False
                for nv in candidates:
                    if nv >= old_val:
                        continue
                    if not check_move_conflict(cur, row_diff_sets, r, pos, nv):
                        continue
                    # Valid move!
                    row[pos] = nv
                    row_diff_sets[r] = update_row_diff_set(row)
                    new_scope = max(max(rr) for rr in cur)
                    if new_scope < cur_scope:
                        cur_scope = new_scope
                        improved = True
                        found = True
                        break
                    elif new_scope == cur_scope:
                        # Accept same-scope move for diversity
                        if rng.random() < 0.3:
                            improved = True
                            found = True
                            break
                    # Revert if worse
                    row[pos] = old_val
                    row_diff_sets[r] = update_row_diff_set(row)

                if found:
                    break

            if not improved:
                # Try a random perturbation to escape local optimum
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else cur_scope + 15
                if hi > lo:
                    nv = rng.randint(lo, hi)
                    if check_move_conflict(cur, row_diff_sets, ri, pos, nv):
                        old = cur[ri][pos]
                        cur[ri][pos] = nv
                        row_diff_sets[ri] = update_row_diff_set(cur[ri])
                        new_scope = max(max(rr) for rr in cur)
                        if new_scope <= cur_scope:
                            cur_scope = new_scope
                            improved = True
                        else:
                            cur[ri][pos] = old
                            row_diff_sets[ri] = update_row_diff_set(cur[ri])

                if not improved:
                    break

        if improved:
            return cur, cur_scope
        return None, None

    # Main search loop: constraint-guided descent with periodic restarts
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            cur = [r[:] for r in best_rows]
        else:
            # Perturb from best
            cur = [r[:] for r in best_rows]
            n_perturb = rng.randint(1, 4)
            for _ in range(n_perturb):
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 15
                if hi > lo:
                    cur[ri][pos] = rng.randint(lo, hi)

        cur_scope = full_validate_scope(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Run constraint-guided descent
        budget = max(0.5, (deadline - time.time()) / max(1, max_restarts - restart))
        budget = min(budget, 3.0)  # Cap per-restart budget
        desc_rows, desc_scope = constraint_guided_search(cur, budget)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    return best_rows
