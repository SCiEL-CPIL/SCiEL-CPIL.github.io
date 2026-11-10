"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start = time.time()
    def now():
        return time.time()
    deadline = start + 140  # safety margin before 150s limit

    diffs_per_row = k * (k + 1) // 2

    def diff_set_of_row(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def validate(rows):
        used = set()
        for r in rows:
            for d in diff_set_of_row(r):
                if d in used:
                    return False
                used.add(d)
        for r in rows:
            if not r or r[0] != 0:
                return False
            for a, b in zip(r, r[1:]):
                if b <= a:
                    return False
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Greedy row builder (baseline) ----------
    def greedy_row(used, temp=0.0, scope=None):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope is not None and x > scope:
                return None, used
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                if temp > 0 and random.random() < temp and x < 300:
                    continue
                row.append(x)
                used = used | new_diffs
        return row, used

    # ---------- Fast backtracking row builder with bitmask ----------
    def build_row_fast(used_mask, scope, time_limit=None, rng=None, random_order=False):
        if time_limit is not None and now() > time_limit:
            return None, None

        target_len = k + 1

        if rng is None:
            rng = random.Random()

        def conflict_mask(row, new_x):
            mask = 0
            for a in row:
                d = new_x - a
                mask |= (1 << d)
            return mask

        def dfs(row, cur_mask):
            if time_limit is not None and now() > time_limit:
                return None
            if len(row) == target_len:
                return row, cur_mask

            last = row[-1]
            rem = target_len - len(row)

            min_next = last + 1
            max_next = scope - (rem - 1)

            if min_next > max_next:
                return None

            candidates = []
            for x in range(min_next, max_next + 1):
                cm = conflict_mask(row, x)
                if not (cm & cur_mask):
                    if x - row[0] <= scope:
                        candidates.append((x, cm))

            if not candidates:
                return None

            if random_order and len(candidates) > 1:
                rng.shuffle(candidates)

            for x, cm in candidates:
                if time_limit is not None and now() > time_limit:
                    return None
                new_mask = cur_mask | cm
                new_row = row + [x]
                res = dfs(new_row, new_mask)
                if res is not None:
                    return res
            return None

        res = dfs([0], used_mask)
        if res is None:
            return None, None
        row, mask = res
        return row, mask

    def used_set_to_mask(used_set, scope):
        mask = 0
        for d in used_set:
            if 0 < d <= scope:
                mask |= (1 << d)
        return mask

    # ---------- Global matching-based solver ----------
    # Model: choose n_rows rows from a pool of candidate rows (each valid within scope,
    # pairwise difference-disjoint). Use max bipartite matching on a graph where
    # left side = row slots, right side = candidate rows, edge if compatible.
    # We build candidates greedily/backtracking, then find max matching.

    def build_candidate_pool(scope, num_candidates, used_pool_mask=None, rng=None, time_limit=None):
        """Generate a pool of candidate rows (each max <= scope) that are internally valid.
        We generate many, keeping only those that don't conflict with each other
        (for the pool to be useful for matching)."""
        if rng is None:
            rng = random.Random()
        pool = []
        pool_diffs = []  # diff sets for quick conflict check
        pool_masks = []  # bitmasks for fast check

        for _ in range(num_candidates):
            if time_limit is not None and now() > time_limit:
                break
            # Try to build a row with random_order for diversity
            row, mask = build_row_fast(0, scope, time_limit=time_limit, rng=rng, random_order=True)
            if row is None:
                continue
            # Check against existing pool
            conflict = False
            for pm in pool_masks:
                if mask & pm:
                    conflict = True
                    break
            if not conflict:
                pool.append(row)
                pool_masks.append(mask)
                pool_diffs.append(diff_set_of_row(row))

        return pool, pool_masks

    def find_max_matching(n_slots, n_cands, adj, rng=None):
        """
        Max bipartite matching via augmenting paths (Hopcroft-Karp simplified).
        adj: list of lists, adj[i] = list of candidate indices compatible with slot i.
        Returns: matching list where matching[i] = candidate index or -1, and
                 matched_cands set.
        """
        if rng is None:
            rng = random.Random()

        match_slot = [-1] * n_slots
        match_cand = [-1] * n_cands

        def try_augment(slot, visited_cands):
            for cand in adj[slot]:
                if cand in visited_cands:
                    continue
                visited_cands.add(cand)
                if match_cand[cand] == -1 or try_augment(match_cand[cand], visited_cands):
                    match_cand[cand] = slot
                    match_slot[slot] = cand
                    return True
            return False

        for s in range(n_slots):
            visited = set()
            try_augment(s, visited)

        return match_slot, match_cand

    def global_search(scope, time_limit, rng, pool_size=50):
        """
        Try to find a valid set of n_rows rows all with max <= scope.
        1. Build a pool of candidate rows (max <= scope, pairwise disjoint).
        2. If pool has >= n_rows rows, find a matching of size n_rows.
        3. Return the matched rows or None.
        """
        if now() > time_limit:
            return None

        pool, pool_masks = build_candidate_pool(scope, pool_size, rng=rng, time_limit=time_limit)

        if len(pool) < n_rows:
            return None

        # Build adjacency: slot i is compatible with candidate c if they're disjoint
        # (they all are, by construction of pool), so any n_rows from pool work.
        # Actually, since pool rows are pairwise disjoint, ANY subset of n_rows works.
        # So we just take the first n_rows.
        # But to be safe and allow for more complex matching later, use the matching framework.
        # For now, since pool is pairwise disjoint, just take n_rows of them.
        if len(pool) >= n_rows:
            # Pick n_rows rows; prefer those with smaller max for tighter packing
            pool_sorted = sorted(range(len(pool)), key=lambda i: pool[i][-1])
            chosen = pool_sorted[:n_rows]
            rows = [pool[i] for i in chosen]
            return rows

        return None

    # ---------- Initial construction ----------
    random.seed(42)
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = greedy_row(used, temp=0.0)
        rows.append(row)
    best_rows = [list(r) for r in rows]
    best_max = max_val(best_rows)

    # ---------- Multi-temperature restarts for better initial ----------
    temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    attempts = 0
    max_attempts = 200
    while now() < deadline - 40 and attempts < max_attempts:
        attempts += 1
        temp = temps[attempts % len(temps)]
        used = set()
        trial_rows = []
        ok = True
        for _ in range(n_rows):
            if now() > deadline - 40:
                ok = False
                break
            row, used = greedy_row(used, temp=temp)
            if row is None:
                ok = False
                break
            trial_rows.append(row)
        if not ok:
            continue
        mv = max_val(trial_rows)
        if mv < best_max:
            best_max = mv
            best_rows = [list(r) for r in trial_rows]

    # ---------- Global matching-based scope search ----------
    # Try to find a better global packing via candidate pool + matching.
    # Iterate scope downward from best_max-1.
    rng_global = random.Random(777)
    for scope in range(best_max - 1, max(best_max - 30, k), -1):
        if now() > deadline - 40:
            break
        if scope < k:
            break
        # Try multiple times with different random seeds for the pool
        found = False
        for attempt in range(5):
            if now() > deadline - 40:
                break
            rows_found = global_search(scope, deadline - 40, rng_global, pool_size=30 + attempt * 10)
            if rows_found is not None:
                if validate(rows_found):
                    best_rows = [list(r) for r in rows_found]
                    best_max = scope
                    found = True
                    break
        if found:
            # Continue trying to go lower
            continue
        else:
            # If we can't find at this scope, try a few more attempts before giving up
            for attempt in range(3):
                if now() > deadline - 40:
                    break
                rows_found = global_search(scope, deadline - 40, rng_global, pool_size=60)
                if rows_found is not None and validate(rows_found):
                    best_rows = [list(r) for r in rows_found]
                    best_max = scope
                    break
            break

    # ---------- Backtracking improvement: replace rows with smaller ones ----------
    def try_replace_row(i, scope_target, time_limit=None, rng=None, random_order=False):
        other_rows = [r for j, r in enumerate(best_rows) if j != i]
        fixed_used = set()
        for r in other_rows:
            fixed_used |= diff_set_of_row(r)

        fixed_mask = used_set_to_mask(fixed_used, scope_target)

        if rng is None:
            rng = random.Random()

        res, _ = build_row_fast(fixed_mask, scope_target, time_limit=time_limit,
                                rng=rng, random_order=random_order)

        if res is not None:
            new_row = res
            new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
            if validate(new_rows):
                return new_row
        return None

    # Iteratively try to lower the scope
    improved = True
    rng = random.Random(999)
    while improved and now() < deadline - 20:
        improved = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if now() > deadline - 20:
                break
            cur_max_row = best_rows[i][-1]
            target = cur_max_row - 1
            if target < k:
                continue
            new_row = try_replace_row(i, target, time_limit=deadline - 20, rng=rng, random_order=True)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                mv = max_val(new_rows)
                if mv < best_max:
                    best_max = mv
                    best_rows = new_rows
                    improved = True

    # ---------- Pair-swap refinement ----------
    if now() < deadline - 30:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        rng.shuffle(pairs)
        for (i, j) in pairs:
            if now() > deadline - 20:
                break
            other_rows = [r for x, r in enumerate(best_rows) if x != i and x != j]
            fixed_used = set()
            for r in other_rows:
                fixed_used |= diff_set_of_row(r)

            for scope in range(best_max - 1, max(best_max - 10, 0), -1):
                if now() > deadline - 20:
                    break
                if scope < k:
                    continue
                fixed_mask = used_set_to_mask(fixed_used, scope)

                r1, mask_after1 = build_row_fast(fixed_mask, scope, time_limit=deadline - 20,
                                                 rng=rng, random_order=True)
                if r1 is None:
                    continue
                r2, _ = build_row_fast(mask_after1, scope, time_limit=deadline - 20,
                                       rng=rng, random_order=True)
                if r2 is None:
                    continue
                new_rows = [r if x not in (i, j) else (r1 if x == i else r2)
                            for x, r in enumerate(best_rows)]
                if validate(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                    break

    # ---------- Final polish: single-row backtracking with tighter scopes ----------
    while now() < deadline - 10:
        any_improve = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if now() > deadline - 10:
                break
            cur_max_row = best_rows[i][-1]
            for target in range(min(cur_max_row - 1, best_max - 1), max(cur_max_row - 8, k) - 1, -1):
                if now() > deadline - 10:
                    break
                if target < k:
                    continue
                new_row = try_replace_row(i, target, time_limit=deadline - 10, rng=rng, random_order=True)
                if new_row is not None:
                    new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                        any_improve = True
                        break
        if not any_improve:
            break

    return best_rows
# EVOLVE-BLOCK-END
