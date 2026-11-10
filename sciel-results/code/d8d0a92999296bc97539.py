"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start = time.time()
    def now():
        return time.time()
    deadline = start + 140  # safety margin before 150s limit

    # Each row has k+1 elements, all differences positive.
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
        """Build one row greedily. scope is an upper bound on the max element."""
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

    # ---------- Fast conflict check using integer bitmask ----------
    def build_row_fast(used_mask, scope, time_limit=None, rng=None, random_order=False):
        """
        Build one row of size k+1 with max <= scope using bitmask for fast conflict checking.
        used_mask: integer whose bit d is 1 if difference d is already used.
        Returns (row, used_mask) or (None, None).
        """
        if time_limit is not None and now() > time_limit:
            return None, None
        
        target_len = k + 1
        
        # Precompute mask of valid differences (0 to scope)
        valid_mask = (1 << (scope + 1)) - 1  # bits 0..scope set
        
        if rng is None:
            rng = random.Random()
        
        def diff_mask_for(row, new_x):
            """Bitmask of all differences new_x - a for a in row (new_x > all a)."""
            mask = 0
            for a in row:
                mask |= (1 << (new_x - a))
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

            # Collect valid candidates (no diff conflict with cur_mask)
            cands = []
            for x in range(min_next, max_next + 1):
                cm = diff_mask_for(row, x)
                if not (cm & cur_mask):
                    cands.append((x, cm))

            if not cands:
                return None

            # Forward-check: remove candidate x if for every y > x
            # (y - x) is already in cur_mask.  Such x can never be
            # extended to a full row.
            if len(cands) > 1:
                filtered = []
                for idx, (x, cm) in enumerate(cands):
                    keep = False
                    for jdx in range(idx + 1, len(cands)):
                        y = cands[jdx][0]
                        if not ((1 << (y - x)) & cur_mask):
                            keep = True
                            break
                    if keep:
                        filtered.append((x, cm))
                cands = filtered
                if not cands:
                    return None

            if random_order and len(cands) > 1:
                rng.shuffle(cands)
            else:
                # Prefer larger gaps (spread elements) to leave room
                cands.sort(key=lambda t: t[0] - last, reverse=True)

            for x, cm in cands:
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

    # ---------- Convert used set to bitmask ----------
    def used_set_to_mask(used_set, scope):
        mask = 0
        for d in used_set:
            if 0 < d <= scope:
                mask |= (1 << d)
        return mask

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
    while now() < deadline - 25 and attempts < max_attempts:
        attempts += 1
        temp = temps[attempts % len(temps)]
        used = set()
        trial_rows = []
        ok = True
        for _ in range(n_rows):
            if now() > deadline - 25:
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

    # ---------- Backtracking improvement: replace rows with smaller ones ----------
    def try_replace_row(i, scope_target, time_limit=None, rng=None, random_order=False):
        """Try to find a valid replacement for row i with max <= scope_target."""
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
