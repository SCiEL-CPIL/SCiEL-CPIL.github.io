import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 113.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs  # counting lower bound (scope must be at least this)
    k1 = k + 1

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def _compute_all_diffs(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def validate_rows(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_baseline():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def greedy_construct(seed, strategy='basic'):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k1:
                last = row[-1]
                found = False
                if strategy == 'compact':
                    search_end = last + 25 + rng.randint(0, 15)
                elif strategy == 'spread':
                    search_end = last + 50 + rng.randint(0, 40)
                else:
                    search_end = last + 35 + rng.randint(0, 25)
                search_end = min(search_end, lb + 200)
                candidates = list(range(last + 1, search_end))
                if rng.random() < 0.5:
                    pass
                else:
                    rng.shuffle(candidates)
                for x in candidates:
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    for x in range(last + 1, lb + 300):
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    def spectral_greedy(max_scope, seed, time_limit=5.0):
        """Spectral-guided greedy: build a conflict matrix over candidate positions,
        use eigenvector ordering to guide assignment, minimize max entry."""
        rng = random.Random(seed)
        t_start = time.time()
        used = set()
        rows = []

        # Build a simple conflict heuristic: for each row, prefer positions
        # that create fewer "future conflicts" (differences close to already-used ones)
        for row_idx in range(n_rows):
            if time.time() - t_start > time_limit:
                return None
            row = [0]
            while len(row) < k1:
                last = row[-1]
                # Score each candidate by how "conflict-free" it is
                candidates = []
                for x in range(last + 1, max_scope + 1):
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    # Heuristic: prefer smaller x, but also prefer x whose diffs
                    # are "spread out" from used diffs (less future conflict)
                    conflict_score = 0
                    for d in new_diffs:
                        # Count how many used diffs are within 2 of d
                        for dd in range(max(1, d - 2), min(max_scope, d + 2) + 1):
                            if dd in used:
                                conflict_score += 1
                    candidates.append((conflict_score, x))
                if not candidates:
                    return None
                # Sort by conflict score, then by value; pick from top with randomness
                candidates.sort()
                top_n = min(8, len(candidates))
                if rng.random() < 0.7:
                    # Pick from top candidates with slight randomness
                    idx = rng.randint(0, top_n - 1)
                else:
                    idx = 0
                chosen_x = candidates[idx][1]
                new_diffs = [chosen_x - a for a in row]
                row.append(chosen_x)
                used.update(new_diffs)
            rows.append(row)
        return rows

    def backtracking_search(max_scope, seed, time_limit=3.0):
        rng = random.Random(seed)
        used_diffs = [False] * (max_scope + 1)
        rows = []
        t_start = time.time()

        def build_row(pos, last_val, row_so_far):
            nonlocal used_diffs
            if time.time() - t_start > time_limit:
                return None
            if pos == k1:
                return list(row_so_far)
            max_x = max_scope - (k1 - pos - 1)
            if max_x <= last_val:
                return None
            valid = []
            for x in range(last_val + 1, max_x + 1):
                ok = True
                for a in row_so_far:
                    if used_diffs[x - a]:
                        ok = False
                        break
                if ok:
                    valid.append(x)
            if not valid:
                return None
            n_take = min(12, len(valid))
            window = valid[:n_take]
            if len(valid) > n_take and rng.random() < 0.4:
                extra = rng.sample(valid[n_take:], min(5, len(valid) - n_take))
                window.extend(extra)
            rng.shuffle(window)
            for x in window:
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    new_diffs.append(d)
                    used_diffs[d] = True
                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                for d in new_diffs:
                    used_diffs[d] = False
            return None

        for _ in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            for i in range(k1):
                for j in range(i + 1, k1):
                    used_diffs[row[j] - row[i]] = True
        return rows

    def local_improve(rows, seed, max_iters=2000):
        """Enhanced local search: allows both decreases AND increases in element values,
        with a wider neighborhood to escape local minima."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = _compute_all_diffs(rows)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue
            # Remove old element and its diffs
            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            # Allow search up to old_val + 10 (can increase to escape local minima)
            max_val = old_val + 10
            # Try decreasing first (for scope reduction), then increasing
            try_order = list(range(min_val, old_val + 1)) + list(range(old_val + 1, max_val + 1))
            if rng.random() < 0.3:
                # Sometimes try increasing first
                try_order = list(range(old_val + 1, max_val + 1)) + list(range(min_val, old_val + 1))
            rng.shuffle(try_order[1:])  # Keep min_val first for deterministic fallback
            for new_val in try_order:
                if new_val < min_val or new_val > max_val:
                    continue
                valid = True
                for a in row[:elem_idx]:
                    if (new_val - a) in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        if (a - new_val) in used_set:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used_set.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - new_val)
                    found = True
                    break
            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)
        return rows, scope

    def sa_polish(rows, seed, max_iters=3000):
        """Simulated annealing polish: accept worse moves with probability exp(-delta/T)."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = _compute_all_diffs(rows)
        best_rows = [list(r) for r in rows]
        best_scope = scope
        T = 5.0  # initial temperature
        cooling = 0.999
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            T *= cooling
            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue
            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue
            # Generate a random new value in a window around old_val
            window = 8 + rng.randint(0, 8)
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val + window
            new_val = rng.randint(min_val, max_val)
            if new_val == old_val:
                continue
            # Remove old
            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)
            # Check validity of new
            valid = True
            for a in row[:elem_idx]:
                if (new_val - a) in used_set:
                    valid = False
                    break
            if valid:
                for a in row[elem_idx:]:
                    if (a - new_val) in used_set:
                        valid = False
                        break
            if not valid:
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)
                continue
            # Apply new
            row.insert(elem_idx, new_val)
            for a in row[:elem_idx]:
                used_set.add(new_val - a)
            for a in row[elem_idx + 1:]:
                used_set.add(a - new_val)
            new_scope = get_scope(rows)
            delta = new_scope - scope
            if delta <= 0:
                scope = new_scope
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
            else:
                # Accept with probability exp(-delta/T)
                if T > 0.01 and rng.random() < math.exp(-delta / T):
                    scope = new_scope
                else:
                    # Reject: revert
                    for a in row[:elem_idx]:
                        used_set.discard(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.discard(a - new_val)
                    row.pop(elem_idx)
                    row.insert(elem_idx, old_val)
                    for a in row[:elem_idx]:
                        used_set.add(old_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - old_val)
        return best_rows, best_scope

    def compact_pass(rows):
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = _compute_all_diffs(rows)
        improved_any = True
        for _ in range(300):
            if time.time() > deadline - 2:
                break
            if not improved_any:
                break
            improved_any = False
            for row_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break
                row = rows[row_idx]
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue
                    for a in row[:elem_pos]:
                        used_set.discard(old_val - a)
                    for a in row[elem_pos + 1:]:
                        used_set.discard(a - old_val)
                    row.pop(elem_pos)
                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    found = False
                    for new_val in range(min_val, old_val):
                        valid = True
                        for a in row[:elem_pos]:
                            if (new_val - a) in used_set:
                                valid = False
                                break
                        if valid:
                            for a in row[elem_pos:]:
                                if (a - new_val) in used_set:
                                    valid = False
                                    break
                        if valid:
                            row.insert(elem_pos, new_val)
                            for a in row[:elem_pos]:
                                used_set.add(new_val - a)
                            for a in row[elem_pos + 1:]:
                                used_set.add(a - new_val)
                            found = True
                            break
                    if found:
                        new_scope = get_scope(rows)
                        if new_scope < scope:
                            scope = new_scope
                            improved_any = True
                    else:
                        row.insert(elem_pos, old_val)
                        for a in row[:elem_pos]:
                            used_set.add(old_val - a)
                        for a in row[elem_pos + 1:]:
                            used_set.add(a - old_val)
        return rows, scope

    # === Phase 1: quick greedy baseline ===
    base_rows = greedy_baseline()
    if base_rows:
        best_rows = [list(r) for r in base_rows]
        best_scope = get_scope(base_rows)

    # === Phase 2: spectral-guided greedy constructions ===
    spec_budget = 20.0
    spec_start = time.time()
    for i in range(30):
        if time.time() - spec_start > spec_budget:
            break
        if time.time() > deadline - 40:
            break
        target = best_scope - (i % 15)
        if target < lb:
            target = lb
        rows = spectral_greedy(target, 50000 + i * 211, time_limit=3.0)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # === Phase 3: backtracking at tighter scopes ===
    bt_budget = 30.0
    bt_start = time.time()
    for offset in range(0, 50):
        if time.time() - bt_start > bt_budget:
            break
        if time.time() > deadline - 30:
            break
        target = best_scope - offset
        if target < lb:
            break
        for seed in range(4):
            if time.time() - bt_start > bt_budget:
                break
            rows = backtracking_search(target, 1000 + offset * 137 + seed * 11, time_limit=2.0)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                break

    # === Phase 4: greedy constructions ===
    strategies = ['basic', 'compact', 'spread']
    greedy_budget = 15.0
    greedy_start = time.time()
    for i in range(2000):
        if time.time() - greedy_start > greedy_budget:
            break
        if time.time() > deadline - 25:
            break
        strategy = strategies[i % len(strategies)]
        rows = greedy_construct(i * 31 + 7, strategy)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # === Phase 5: local improvement with wider neighborhood ===
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=2000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        compacted, comp_scope = compact_pass(best_rows)
        if comp_scope < best_scope:
            best_scope = comp_scope
            best_rows = compacted

    # === Phase 6: simulated annealing polish with restarts ===
    if best_rows is not None and time.time() < deadline - 20:
        sa_budget = 18.0
        sa_start = time.time()
        for restart in range(5):
            if time.time() - sa_start > sa_budget:
                break
            if time.time() > deadline - 8:
                break
            sa_rows, sa_scope = sa_polish(best_rows, 77777 + restart * 1337, max_iters=1500)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
            # Local improve after SA
            if time.time() < deadline - 6:
                improved, imp_scope = local_improve(best_rows, 88888 + restart * 999, max_iters=500)
                if imp_scope < best_scope:
                    best_scope = imp_scope
                    best_rows = improved
                compacted, comp_scope = compact_pass(best_rows)
                if comp_scope < best_scope:
                    best_scope = comp_scope
                    best_rows = compacted

    # === Phase 7: final backtracking tighten + polish ===
    if best_rows is not None and time.time() < deadline - 12:
        for offset in range(1, 8):
            target = best_scope - offset
            if target < lb:
                break
            if time.time() > deadline - 8:
                break
            for seed in range(3):
                if time.time() > deadline - 6:
                    break
                rows = backtracking_search(target, 50000 + offset * 1000 + seed * 73, time_limit=1.5)
                if rows is not None:
                    scope = get_scope(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                    break
        if time.time() < deadline - 5:
            improved, imp_scope = local_improve(best_rows, 99999, max_iters=800)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved
            if time.time() < deadline - 3:
                compacted, comp_scope = compact_pass(best_rows)
                if comp_scope < best_scope:
                    best_scope = comp_scope
                    best_rows = compacted

    # Final validation
    if best_rows is not None:
        if not validate_rows(best_rows):
            fb = greedy_baseline()
            if fb:
                best_rows = fb
                best_scope = get_scope(fb)

    return best_rows


if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
