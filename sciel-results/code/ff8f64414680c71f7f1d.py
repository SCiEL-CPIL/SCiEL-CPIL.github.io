import time
import math
import random
import sys

def construct(n_rows, k):
    """
    Construct difference triangle set (n_rows, k).
    Returns a list of n_rows rows, each of length k+1, starting at 0, strictly increasing.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit

    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    M_lower = total_diffs_needed

    # ---- Greedy construction for initial upper bound ----
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    greedy_rows = greedy_construct()
    best_scope = max(max(r) for r in greedy_rows)
    best_rows = [r[:] for r in greedy_rows]

    # ---- Smooth surrogate objective machinery ----
    # We track diff_counts[d] = number of times difference d appears.
    # Smooth energy = sum_d diff_counts[d]^2  (minimized when all counts are 0 or 1).
    # Plus a soft scope penalty via auxiliary variable t (epigraph):
    #   energy = sum_d c[d]^2 + lam * max(0, max_entry - t)^2
    # We optimize t implicitly by just using current scope; the key change is
    # the squared surrogate instead of the 0/1 duplicate indicator.

    def build_diff_counts(rows, max_val):
        counts = [0] * (max_val + 1)
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d <= max_val:
                        counts[d] += 1
        return counts

    def energy_from_counts(counts, scope, t, lam):
        # sum of squares
        s = 0
        for c in counts:
            if c > 1:
                s += c * c
            # c<=1 contributes c^2 = c, but we only care about >1 for conflicts;
            # actually sum c^2 over all: c=1 -> 1, c=0 -> 0. We include all for smoothness.
        # Recompute properly:
        s = 0
        for c in counts:
            s += c * c
        # scope penalty: (max(0, scope - t))^2
        if scope > t:
            s += lam * (scope - t) ** 2
        return s

    def get_scope(rows):
        return max(max(r) for r in rows)

    # ---- Phase 1: Conflict-driven targeted local search (as in current design) ----
    def compute_all_diffs(rows):
        diff_map = {}
        for ri, row in enumerate(rows):
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d not in diff_map:
                        diff_map[d] = []
                    diff_map[d].append((ri, i, j))
        return diff_map

    def count_conflicts(diff_map):
        return sum(1 for v in diff_map.values() if len(v) > 1)

    def targeted_backtrack(rows, max_scope, time_limit):
        rows = [r[:] for r in rows]
        diff_map = compute_all_diffs(rows)
        n_conflicts = count_conflicts(diff_map)

        if n_conflicts == 0:
            return rows, 0

        elem_conflict_count = {}

        def update_elem_conflicts(diff_map):
            elem_conflict_count.clear()
            for d, positions in diff_map.items():
                if len(positions) > 1:
                    for (ri, i, j) in positions:
                        key1 = (ri, i)
                        key2 = (ri, j)
                        elem_conflict_count[key1] = elem_conflict_count.get(key1, 0) + 1
                        elem_conflict_count[key2] = elem_conflict_count.get(key2, 0) + 1
            return elem_conflict_count

        elem_conflict_count = update_elem_conflicts(diff_map)

        best_local = [r[:] for r in rows]
        best_local_conflicts = n_conflicts

        max_iter = 15000
        iter_count = 0

        while iter_count < max_iter and time.time() < time_limit:
            if n_conflicts == 0:
                break

            if not elem_conflict_count:
                break

            max_conf = max(elem_conflict_count.values())
            candidates = [key for key, cnt in elem_conflict_count.items() if cnt == max_conf]
            ri, ei = random.choice(candidates)

            row = rows[ri]
            old_val = row[ei]

            lower = row[ei - 1] + 1 if ei > 0 else 1
            upper = row[ei + 1] - 1 if ei < k else max_scope

            if lower > upper:
                if ei == k:
                    upper = min(max_scope + 5, best_scope + 20)
                else:
                    iter_count += 1
                    continue

            num_tries = min(20, upper - lower + 1)
            best_new_val = old_val
            best_new_conflicts = n_conflicts

            for _ in range(num_tries):
                new_val = random.randint(lower, upper)
                if new_val == old_val:
                    continue

                saved_row = row[:]
                row[ei] = new_val
                new_diff_map = compute_all_diffs(rows)
                new_conflicts = count_conflicts(new_diff_map)
                row[:] = saved_row

                if new_conflicts < best_new_conflicts:
                    best_new_conflicts = new_conflicts
                    best_new_val = new_val

            if best_new_val != old_val and best_new_conflicts < n_conflicts:
                row[ei] = best_new_val
                diff_map = compute_all_diffs(rows)
                n_conflicts = best_new_conflicts
                elem_conflict_count = update_elem_conflicts(diff_map)

                if n_conflicts == 0:
                    best_local = [r[:] for r in rows]
                    break
                elif n_conflicts < best_local_conflicts:
                    best_local_conflicts = n_conflicts
                    best_local = [r[:] for r in rows]
            else:
                if random.random() < 0.3:
                    ri2 = random.randint(0, n_rows - 1)
                    ei2 = random.randint(1, k)
                    row2 = rows[ri2]
                    lower2 = row2[ei2 - 1] + 1
                    upper2 = row2[ei2 + 1] - 1 if ei2 < k else max_scope
                    if lower2 <= upper2:
                        new_val2 = random.randint(lower2, upper2)
                        if new_val2 != row2[ei2]:
                            row2[ei2] = new_val2
                            diff_map = compute_all_diffs(rows)
                            n_conflicts = count_conflicts(diff_map)
                            elem_conflict_count = update_elem_conflicts(diff_map)

                            if n_conflicts < best_local_conflicts:
                                best_local_conflicts = n_conflicts
                                best_local = [r[:] for r in rows]

            iter_count += 1

        return best_local, best_local_conflicts

    # Multi-start targeted backtracking to reduce scope
    for target_scope in range(best_scope - 1, max(M_lower - 1, best_scope - 30), -1):
        if time.time() > deadline - 20:
            break

        for attempt in range(4):
            if time.time() > deadline - 20:
                break

            start_rows = [r[:] for r in best_rows]

            num_perturb = random.randint(3, 8)
            for _ in range(num_perturb):
                ri = random.randint(0, n_rows - 1)
                ei = random.randint(1, k)
                row = start_rows[ri]
                lower = row[ei - 1] + 1
                upper = row[ei + 1] - 1 if ei < k else target_scope
                if lower <= upper:
                    row[ei] = random.randint(lower, upper)

            result_rows, result_conflicts = targeted_backtrack(
                start_rows, target_scope, deadline - 20
            )

            if result_conflicts == 0:
                new_scope = max(max(r) for r in result_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in result_rows]
                break

    # ---- Phase 2: Smooth-surrogate Simulated Annealing ----
    # Energy = sum_d c[d]^2 + lam * max(0, scope - t)^2
    # This is a smooth surrogate for the nonsmooth "number of duplicate diffs" objective.
    # The squared term gives gradient-like pressure toward resolving conflicts,
    # unlike the 0/1 indicator which has flat plateaus.

    if time.time() < deadline - 20:
        current_rows = [r[:] for r in best_rows]
        cur_scope = get_scope(current_rows)

        max_val = best_scope + 80
        diff_counts = build_diff_counts(current_rows, max_val)

        # Auxiliary epigraph variable t: target scope. Start at current scope,
        # gradually decrease to encourage tighter packing.
        t = float(best_scope)
        t_target = float(M_lower)  # lower bound
        lam = 0.5  # weight for scope penalty

        def current_energy():
            s = 0
            for c in diff_counts:
                s += c * c
            scope = get_scope(current_rows)
            if scope > t:
                s += lam * (scope - t) ** 2
            return s

        cur_energy = current_energy()

        T = 10.0
        alpha = 0.9995
        min_T = 0.005
        steps = 0
        max_steps = 200000

        while steps < max_steps and time.time() < deadline - 10:
            # Gradually tighten t toward target
            t = t + (t_target - t) * 0.001
            if t < t_target:
                t = t_target

            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)
            row = current_rows[ri]
            old_val = row[ei]

            lower = row[ei - 1] + 1
            upper = row[ei + 1] - 1 if ei < k else int(t) + 5

            if lower > upper:
                steps += 1
                continue

            new_val = random.randint(lower, upper)
            if new_val == old_val:
                steps += 1
                continue

            # Compute delta in smooth energy
            delta = 0
            for j in range(len(row)):
                if j == ei:
                    continue
                d_old = abs(row[j] - old_val)
                if d_old <= max_val:
                    c_old = diff_counts[d_old]
                    delta -= (c_old * c_old)
                    diff_counts[d_old] = c_old - 1
                    delta += ((c_old - 1) * (c_old - 1))

                d_new = abs(row[j] - new_val)
                if d_new <= max_val:
                    c_new = diff_counts[d_new]
                    delta += (c_new + 1) * (c_new + 1) - c_new * c_new
                    diff_counts[d_new] = c_new + 1

            # Scope penalty delta
            old_scope = get_scope(current_rows)
            # temporarily apply to get new scope
            row[ei] = new_val
            new_scope = get_scope(current_rows)
            row[ei] = old_val  # revert for now

            old_pen = lam * (old_scope - t) ** 2 if old_scope > t else 0.0
            new_pen = lam * (new_scope - t) ** 2 if new_scope > t else 0.0
            delta += (new_pen - old_pen)

            # Acceptance
            if delta < 0:
                accept = True
            else:
                if T > min_T:
                    accept = random.random() < math.exp(-delta / T)
                else:
                    accept = False

            if accept:
                # Re-apply (we already updated diff_counts above)
                row[ei] = new_val
                cur_energy += delta
                cur_scope = new_scope

                # Track best valid solution
                # Check if all diffs are unique (conflict-free)
                # Quick check: if max count <= 1
                # We'll do a full check periodically
                if steps % 500 == 0:
                    # Validate
                    used = set()
                    valid = True
                    for r in current_rows:
                        n = len(r)
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = r[j] - r[i]
                                if d in used:
                                    valid = False
                                    break
                                used.add(d)
                        if not valid:
                            break
                    if valid:
                        sc = get_scope(current_rows)
                        if sc < best_scope:
                            best_scope = sc
                            best_rows = [r[:] for r in current_rows]

            else:
                # Revert diff_counts
                for j in range(len(row)):
                    if j == ei:
                        continue
                    d_old = abs(row[j] - old_val)
                    if d_old <= max_val:
                        c = diff_counts[d_old]
                        diff_counts[d_old] = c + 1
                    d_new = abs(row[j] - new_val)
                    if d_new <= max_val:
                        c = diff_counts[d_new]
                        diff_counts[d_new] = c - 1

            T *= alpha
            if T < min_T:
                T = min_T
            steps += 1

        # Final check of current_rows
        used = set()
        valid = True
        for r in current_rows:
            n = len(r)
            for i in range(n):
                for j in range(i + 1, n):
                    d = r[j] - r[i]
                    if d in used:
                        valid = False
                        break
                    used.add(d)
            if not valid:
                break
        if valid:
            sc = get_scope(current_rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in current_rows]

    # ---- Phase 3: Exact duplicate-elimination polish ----
    # If best_rows has any remaining duplicates (shouldn't if valid), do greedy repair.
    # Also try to tighten scope by moving the max element down if possible.
    if time.time() < deadline - 5:
        used_diffs = set()
        for row in best_rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    used_diffs.add(row[j] - row[i])

        # Try to reduce scope: for the row with the max entry, try moving it down
        max_row_idx = 0
        max_val_found = -1
        for ri, row in enumerate(best_rows):
            if row[-1] > max_val_found:
                max_val_found = row[-1]
                max_row_idx = ri

        for new_max in range(max_val_found - 1, max_val_found - 20, -1):
            if time.time() > deadline - 5:
                break
            row = best_rows[max_row_idx]
            old_max = row[-1]
            # Check if moving last element to new_max keeps all diffs unique
            # and strictly increasing
            if new_max <= row[-2]:
                continue
            # Remove old diffs involving last element
            temp_used = set(used_diffs)
            for i in range(len(row) - 1):
                d = old_max - row[i]
                temp_used.discard(d)
            # Add new diffs
            ok = True
            for i in range(len(row) - 1):
                d = new_max - row[i]
                if d in temp_used:
                    ok = False
                    break
                temp_used.add(d)
            if ok:
                row[-1] = new_max
                used_diffs = temp_used
                best_scope = new_max
                break

    # Final validation
    used_diffs = set()
    valid = True
    for row in best_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used_diffs:
                    valid = False
                    break
                used_diffs.add(d)
        if not valid:
            break

    if valid:
        return best_rows
    else:
        return greedy_rows
