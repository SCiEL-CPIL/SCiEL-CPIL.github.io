"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start = time.time()
    def now():
        return time.time()
    deadline = start + 145  # safety margin before 150s limit

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

    # ---------- Bitset-based backtracking row builder ----------
    def backtrack_row_bitset(used_set, scope, time_limit=None, random_order=False, seed=0):
        """
        Build one row of size k+1 with max <= scope using bitset-based conflict checking.
        Returns (row, used_set) or (None, None).
        """
        if time_limit is not None and now() > time_limit:
            return None, None
        
        target_len = k + 1
        # Convert used set to a bitset (list of booleans) for fast lookup
        # Max possible difference is scope
        used_bits = bytearray(scope + 1)
        for d in used_set:
            if d <= scope:
                used_bits[d] = 1
        
        rng = random.Random(seed)
        
        def dfs(row, cur_used_bits):
            if time_limit is not None and now() > time_limit:
                return None
            if len(row) == target_len:
                return row, cur_used_bits
            
            last = row[-1]
            rem = target_len - len(row)
            
            min_next = last + 1
            max_next = scope - (rem - 1)
            
            if min_next > max_next:
                return None
            
            # Collect valid candidates
            candidates = []
            for x in range(min_next, max_next + 1):
                # Check conflicts using bitset
                conflict = False
                for a in row:
                    d = x - a
                    if d <= scope and cur_used_bits[d]:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
            
            if not candidates:
                return None
            
            # Shuffle candidates if random_order
            if random_order and len(candidates) > 1:
                rng.shuffle(candidates)
            
            for x in candidates:
                if time_limit is not None and now() > time_limit:
                    return None
                
                # Update used bits
                new_bits = bytearray(cur_used_bits)
                for a in row:
                    d = x - a
                    if d <= scope:
                        new_bits[d] = 1
                
                new_row = row + [x]
                res = dfs(new_row, new_bits)
                if res is not None:
                    return res
            return None

        res = dfs([0], used_bits)
        if res is None:
            return None, None
        row, bits = res
        # Reconstruct used set from bits
        new_used = set(used_set)
        for d in range(1, scope + 1):
            if bits[d]:
                new_used.add(d)
        return row, new_used

    # ---------- Original backtracking (kept for compatibility) ----------
    def backtrack_row(used, scope, time_limit=None):
        """
        Build one row of size k+1 with max <= scope, all diffs disjoint from used.
        Returns (row, used_set) or (None, None). Uses DFS with aggressive pruning.
        """
        if time_limit is not None and now() > time_limit:
            return None, None
        
        row = [0]
        cur_used = set(used)
        target_len = k + 1
        
        def dfs(row, cur_used):
            if time_limit is not None and now() > time_limit:
                return None
            if len(row) == target_len:
                return row, cur_used
            
            last = row[-1]
            rem = target_len - len(row)
            
            min_next = last + 1
            max_next = scope - (rem - 1)
            
            if min_next > max_next:
                return None
            
            for x in range(min_next, max_next + 1):
                if time_limit is not None and now() > time_limit:
                    return None
                
                conflict = False
                for a in row:
                    d = x - a
                    if d in cur_used:
                        conflict = True
                        break
                if conflict:
                    continue
                
                new_row = row + [x]
                new_cur = cur_used.copy()
                for a in row:
                    new_cur.add(x - a)
                
                res = dfs(new_row, new_cur)
                if res is not None:
                    return res
            return None

        res = dfs(row, cur_used)
        if res is None:
            return None, None
        return res

    # ---------- Initial construction ----------
    random.seed(1234)
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = greedy_row(used, temp=0.0)
        rows.append(row)
    best_rows = [list(r) for r in rows]
    best_max = max_val(best_rows)

    # ---------- Multi-temperature restarts for better initial ----------
    temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    attempts = 0
    while now() < deadline - 20 and attempts < 150:
        attempts += 1
        temp = temps[attempts % len(temps)]
        used = set()
        trial_rows = []
        ok = True
        for _ in range(n_rows):
            if now() > deadline - 20:
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
    def try_replace_row(i, scope_target, time_limit=None, use_bitset=True, random_order=False, seed=0):
        """Try to find a valid replacement for row i with max <= scope_target."""
        other_rows = [r for j, r in enumerate(best_rows) if j != i]
        fixed_used = set()
        for r in other_rows:
            fixed_used |= diff_set_of_row(r)
        
        if use_bitset:
            res, _ = backtrack_row_bitset(fixed_used, scope_target, time_limit=time_limit, 
                                          random_order=random_order, seed=seed)
        else:
            res, _ = backtrack_row(fixed_used, scope_target, time_limit=time_limit)
        
        if res is not None:
            new_row = res
            new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
            if validate(new_rows):
                return new_row
        return None

    # Iteratively try to lower the scope
    improved = True
    while improved and now() < deadline - 15:
        improved = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if now() > deadline - 15:
                break
            cur_max_row = best_rows[i][-1]
            target = cur_max_row - 1
            new_row = try_replace_row(i, target, time_limit=deadline - 15, use_bitset=True)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                mv = max_val(new_rows)
                if mv < best_max:
                    best_max = mv
                    best_rows = new_rows
                    improved = True

    # ---------- Pair-swap refinement ----------
    if now() < deadline - 25:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        random.shuffle(pairs)
        for (i, j) in pairs:
            if now() > deadline - 15:
                break
            other_rows = [r for x, r in enumerate(best_rows) if x != i and x != j]
            fixed_used = set()
            for r in other_rows:
                fixed_used |= diff_set_of_row(r)
            
            for scope in range(best_max - 1, max(best_max - 8, 0), -1):
                if now() > deadline - 15:
                    break
                r1, used_after1 = backtrack_row_bitset(fixed_used, scope, time_limit=deadline - 15)
                if r1 is None:
                    continue
                r2, _ = backtrack_row_bitset(used_after1, scope, time_limit=deadline - 15)
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
    while now() < deadline - 5:
        any_improve = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if now() > deadline - 5:
                break
            cur_max_row = best_rows[i][-1]
            for target in range(min(cur_max_row - 1, best_max - 1), max(cur_max_row - 6, 0) - 1, -1):
                if now() > deadline - 5:
                    break
                new_row = try_replace_row(i, target, time_limit=deadline - 5, use_bitset=True)
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
