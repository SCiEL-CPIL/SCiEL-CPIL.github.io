import time
import random
import math
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit

    def row_diffs(row):
        out = set()
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                out.add(row[j] - a)
        return out

    def greedy_row(used_set, k, cap=None, randomize=False, rng=None):
        if rng is None:
            rng = random
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    def dfs_row(row, used_set, cap, row_deadline, randomize=False, rng=None):
        if time.time() > row_deadline or time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row[:]
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)
        if upper < last + 1:
            return None

        candidates = list(range(last + 1, upper + 1))
        if randomize and rng is not None:
            rng.shuffle(candidates)

        for x in candidates:
            if time.time() > row_deadline or time.time() > deadline:
                return None
            new = {x - a for a in row}
            if new & used_set:
                continue
            res = dfs_row(row + [x], used_set | new, cap, row_deadline, randomize, rng)
            if res is not None:
                return res
        return None

    def validate_solution(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            diffs = row_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True

    # Phase 1: Greedy Baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        if row is None:
            row = greedy_row(set(used), k, randomize=True)
        if row is None:
            return [[0] * (k + 1) for _ in range(n_rows)]
        rows.append(row)
        used |= row_diffs(row)

    current_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = current_max
    row_diff_sets = [row_diffs(r) for r in rows]

    # Phase 2: Simulated Annealing
    T = 25.0
    cooling = 0.993
    rng = random.Random(42)

    def generate_candidate(i, cap=None, randomize=True):
        if cap is None:
            cap = current_max
        used_without_i = set(used)
        used_without_i.difference_update(row_diff_sets[i])
        row_deadline = min(deadline - 5, time.time() + 0.4)
        cand = dfs_row([0], used_without_i, cap, row_deadline, randomize=randomize, rng=rng)
        return cand

    no_improve_count = 0
    while time.time() < deadline - 30:
        if T < 0.05:
            break

        i = rng.randint(0, n_rows - 1)
        cap = current_max
        if rng.random() < 0.2:
            cap = current_max + rng.randint(1, 10)

        cand = generate_candidate(i, cap=cap, randomize=True)

        if cand is not None:
            new_last = cand[-1]
            max_others = 0
            for j in range(n_rows):
                if j != i and rows[j][-1] > max_others:
                    max_others = rows[j][-1]

            new_global_max = max(max_others, new_last)
            delta = new_global_max - current_max

            if delta < 0:
                accept = True
            else:
                try:
                    prob = math.exp(-delta / T)
                except:
                    prob = 0.0
                accept = rng.random() < prob

            if accept:
                used.difference_update(row_diff_sets[i])
                rows[i] = cand
                row_diff_sets[i] = row_diffs(cand)
                used |= row_diff_sets[i]
                current_max = new_global_max

                if current_max < best_max:
                    best_max = current_max
                    best_rows = rows[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > 200:
                    no_improve_count = 0
                    # Multi-row restart: regenerate 2-3 random rows
                    for _ in range(rng.randint(2, 3)):
                        j = rng.randint(0, n_rows - 1)
                        used.difference_update(row_diff_sets[j])
                        new_row = greedy_row(set(used), k, randomize=True, rng=rng)
                        if new_row is not None:
                            rows[j] = new_row
                            row_diff_sets[j] = row_diffs(new_row)
                            used |= row_diff_sets[j]
                            current_max = max(r[-1] for r in rows)

        T *= cooling

    # Phase 3: Intensive local search on best_rows
    final_rows = best_rows[:]
    final_used = set()
    final_row_diffs = []
    for r in final_rows:
        d = row_diffs(r)
        final_used |= d
        final_row_diffs.append(d)

    # 3a: Try to shrink each row via binary search on cap
    for i in range(n_rows):
        if time.time() > deadline - 15:
            break
        cur_last = final_rows[i][-1]
        if cur_last == 0:
            continue

        used_without = set(final_used)
        used_without.difference_update(final_row_diffs[i])

        found = None
        low_cap = 0
        high_cap = cur_last - 1

        row_deadline = min(deadline - 15, time.time() + 1.0)
        test_cand = dfs_row([0], used_without, high_cap, row_deadline, randomize=False, rng=rng)
        if test_cand is None:
            continue

        while low_cap <= high_cap and time.time() < deadline - 15:
            mid_cap = (low_cap + high_cap) // 2
            row_deadline = min(deadline - 15, time.time() + 0.8)
            cand = dfs_row([0], used_without, mid_cap, row_deadline, randomize=False, rng=rng)
            if cand is not None:
                found = cand
                high_cap = mid_cap - 1
            else:
                low_cap = mid_cap + 1

        if found is not None:
            final_used.difference_update(final_row_diffs[i])
            final_rows[i] = found
            final_row_diffs[i] = row_diffs(found)
            final_used |= final_row_diffs[i]

    # 3b: Pairwise row swap attempts
    if time.time() < deadline - 10:
        for i in range(n_rows):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, n_rows):
                if time.time() > deadline - 10:
                    break
                old_i = final_rows[i]
                old_j = final_rows[j]

                used_without_i = set(final_used)
                used_without_i.difference_update(final_row_diffs[i])
                used_without_j = set(final_used)
                used_without_j.difference_update(final_row_diffs[j])

                # Try to regenerate row i using space freed by removing both
                combined_free = set(final_used)
                combined_free.difference_update(final_row_diffs[i])
                combined_free.difference_update(final_row_diffs[j])

                cap_i = min(old_i[-1], old_j[-1])
                row_deadline = min(deadline - 10, time.time() + 0.3)
                new_i = dfs_row([0], combined_free, cap_i, row_deadline, randomize=False, rng=rng)

                if new_i is not None and new_i[-1] < old_i[-1]:
                    final_used.difference_update(final_row_diffs[i])
                    final_rows[i] = new_i
                    final_row_diffs[i] = row_diffs(new_i)
                    final_used |= final_row_diffs[i]

    # 3c: Final tight polish - try each row with cap = current global max - 1
    if time.time() < deadline - 5:
        current_global = max(r[-1] for r in final_rows)
        for i in range(n_rows):
            if time.time() > deadline - 5:
                break
            if final_rows[i][-1] <= current_global - 1:
                continue
            used_without = set(final_used)
            used_without.difference_update(final_row_diffs[i])
            row_deadline = min(deadline - 5, time.time() + 0.5)
            cand = dfs_row([0], used_without, current_global - 1, row_deadline, randomize=False, rng=rng)
            if cand is not None:
                final_used.difference_update(final_row_diffs[i])
                final_rows[i] = cand
                final_row_diffs[i] = row_diffs(cand)
                final_used |= final_row_diffs[i]

    # Final validation
    if not validate_solution(final_rows):
        if validate_solution(best_rows):
            return best_rows
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(set(used), k)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return final_rows
