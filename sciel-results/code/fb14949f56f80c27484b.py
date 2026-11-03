import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs  # counting lower bound
    k1 = k + 1

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

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
        """Fast greedy: smallest valid x each step."""
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
        """Construct rows using greedy with randomization."""
        rng = random.Random(seed)
        used = set()
        rows = []

        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k1:
                last = row[-1]
                found = False

                if strategy == 'compact':
                    search_end = last + 20 + rng.randint(0, 15)
                elif strategy == 'spread':
                    search_end = last + 50 + rng.randint(0, 40)
                else:
                    search_end = last + 30 + rng.randint(0, 20)

                search_end = min(search_end, lb + 150)

                candidates = list(range(last + 1, search_end))
                if rng.random() < 0.5:
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
                    for x in range(last + 1, lb + 200):
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

    def backtracking_search(max_scope, seed, time_limit=5.0):
        """
        Backtracking with forward checking and MRV heuristic.
        """
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

            # Prune: need (k1 - pos) more elements, each >= 1 apart
            max_x = max_scope - (k1 - pos - 1)
            if max_x <= last_val:
                return None

            # MRV: find candidates that leave the most options for next position
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

            # Bias toward smaller values for compactness, but sample for diversity
            n_take = min(15, len(valid))
            window = valid[:n_take]
            if len(valid) > n_take and rng.random() < 0.5:
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

    def rebuild_row(row_idx, rows, used_set, max_scope, seed):
        """Rebuild a single row to reduce its maximum value."""
        rng = random.Random(seed)
        row = rows[row_idx]
        old_row = list(row)
        old_max = max(row)

        # Remove this row's diffs from used_set
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_set.discard(row[j] - row[i])

        # Try to rebuild with smaller max
        target_max = old_max - rng.randint(1, 10)
        if target_max < k:  # minimum possible
            target_max = k

        # Backtrack to build new row within target_max
        def build_new_row(pos, last_val, row_so_far):
            if pos == k1:
                return list(row_so_far)

            max_x = min(target_max, max_scope - (k1 - pos - 1))
            if max_x <= last_val:
                return None

            valid = []
            for x in range(last_val + 1, max_x + 1):
                ok = True
                for a in row_so_far:
                    if (x - a) in used_set:
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
                new_diffs = [x - a for a in row_so_far]
                # Check all new diffs are available
                if any(d in used_set for d in new_diffs):
                    continue
                # Temporarily mark
                for d in new_diffs:
                    used_set.add(d)
                res = build_new_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res
                # Undo
                for d in new_diffs:
                    used_set.discard(d)
            return None

        new_row = build_new_row(1, 0, [0])
        if new_row is not None and max(new_row) < old_max:
            # Add new diffs to used_set
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    used_set.add(new_row[j] - new_row[i])
            rows[row_idx] = new_row
            return True
        else:
            # Restore old row
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    used_set.add(old_row[j] - old_row[i])
            rows[row_idx] = old_row
            return False

    def local_improve(rows, seed, max_iters=3000):
        """Try to reduce scope by rebuilding rows."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)

        used_set = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_set.add(row[j] - row[i])

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break

            row_idx = rng.randint(0, n_rows - 1)
            if max(rows[row_idx]) <= k:
                continue

            if rebuild_row(row_idx, rows, used_set, scope, rng.randint(0, 100000)):
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope

        return rows, scope

    def compact_pass(rows):
        """Systematic compaction: repeatedly try to shift each element to its smallest valid position."""
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_set.add(row[j] - row[i])

        improved_any = True
        max_rounds = 200

        for _ in range(max_rounds):
            if time.time() > deadline - 3:
                break
            if not improved_any:
                break
            improved_any = False

            for row_idx in range(n_rows):
                if time.time() > deadline - 3:
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
                            d = new_val - a
                            if d in used_set:
                                valid = False
                                break
                        if valid:
                            for a in row[elem_pos:]:
                                d = a - new_val
                                if d in used_set:
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

    # === Phase 1: Quick greedy baseline ===
    base_rows = greedy_baseline()
    if base_rows:
        base_scope = get_scope(base_rows)
        best_rows = [list(r) for r in base_rows]
        best_scope = base_scope

    # === Phase 2: Backtracking search at various target scopes ===
    target_scopes = []
    if best_scope < float('inf'):
        for s in range(best_scope, lb - 1, -1):
            target_scopes.append(s)
    else:
        target_scopes = [lb + 50, lb + 30, lb + 20, lb + 10, lb]

    bt_budget = 50.0
    bt_start = time.time()
    for target in target_scopes:
        if time.time() - bt_start > bt_budget:
            break
        if target < lb:
            break
        for seed in range(8):
            if time.time() - bt_start > bt_budget:
                break
            if time.time() > deadline - 40:
                break
            rows = backtracking_search(target, 1000 + seed * 137, time_limit=3.0)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    break

    # === Phase 3: Greedy constructions ===
    strategies = ['basic', 'compact', 'spread']
    greedy_budget = 20.0
    greedy_start = time.time()
    for i in range(2000):
        if time.time() - greedy_start > greedy_budget:
            break
        if time.time() > deadline - 40:
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

    # === Phase 4: Local improvement (row rebuild) ===
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=2000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # === Phase 5: More backtracking at tighter scopes ===
    if best_rows is not None and time.time() < deadline - 20:
        for offset in range(1, 15):
            target = best_scope - offset
            if target < lb:
                break
            if time.time() > deadline - 15:
                break
            for seed in range(4):
                if time.time() > deadline - 12:
                    break
                rows = backtracking_search(target, 50000 + offset * 1000 + seed * 73, time_limit=2.0)
                if rows is not None:
                    scope = get_scope(rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [list(r) for r in rows]
                    break

    # === Phase 6: Final local improvement + compaction ===
    if best_rows is not None and time.time() < deadline - 8:
        improved, imp_scope = local_improve(best_rows, 99999, max_iters=1500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

        if time.time() < deadline - 5:
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
