import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    total_diffs = n_rows * (k + 1) * k // 2

    def _valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    if r[j] <= r[i]:
                        return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def _scope(rows):
        return max(max(r) for r in rows)

    def _diffs_of_row(r):
        diffs = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diffs.add(r[j] - r[i])
        return diffs

    def _interleaved_greedy(M_limit, seed_val=0):
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        local_diffs = [set() for _ in range(n_rows)]

        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break

            target = min(candidates, key=lambda idx: rows[idx][-1])
            row = rows[target]
            ld = local_diffs[target]

            x = row[-1] + 1
            found = False
            while x <= M_limit:
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in ld:
                        bad = True
                        break
                    new_diffs.append(d)

                if not bad:
                    row.append(x)
                    ld.update(new_diffs)
                    used_diffs.update(new_diffs)
                    found = True
                    break
                x += 1

            if not found:
                return None

        for r in rows:
            if len(r) != k + 1:
                return None
        return rows

    def _sequential_greedy(M_limit, seed_val=0, skip_prob=0.0):
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            local_diffs = set()
            while len(row) < k + 1:
                x = row[-1] + 1
                found = False
                while x <= M_limit:
                    if rng.random() < skip_prob and x < M_limit:
                        x += 1
                        continue
                    bad = False
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in local_diffs:
                            bad = True
                            break
                        new_diffs.append(d)
                    if not bad:
                        row.append(x)
                        local_diffs.update(new_diffs)
                        found = True
                        break
                    x += 1
                if not found:
                    return None
            rows.append(row)
            for d in local_diffs:
                used_diffs.add(d)

        return rows

    def _repair_row(used_without, max_val, rng, seed_val=0):
        """Build a single row of length k+1 starting at 0 with diffs not in used_without."""
        r2 = random.Random(seed_val)
        row = [0]
        local_diffs = set()
        while len(row) < k + 1:
            x = row[-1] + 1
            found = False
            while x <= max_val:
                if r2.random() < 0.15 and x < max_val:
                    x += 1
                    continue
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_without or d in local_diffs:
                        bad = True
                        break
                    new_diffs.append(d)
                if not bad:
                    row.append(x)
                    local_diffs.update(new_diffs)
                    found = True
                    break
                x += 1
            if not found:
                return None
        return row

    def _replace_row(rows, ri, used_diffs, max_val, rng, seed_val=0):
        old_row = rows[ri]
        old_diffs = _diffs_of_row(old_row)
        used_without = used_diffs - old_diffs

        for attempt in range(30):
            new_row = _repair_row(used_without, max_val, rng, seed_val + attempt)
            if new_row is not None:
                new_diffs = _diffs_of_row(new_row)
                if not (new_diffs & used_without):
                    rows[ri] = new_row
                    used_diffs = used_without | new_diffs
                    return True
        return False

    def _hill_climb(rows, time_budget, seed_val=0):
        if rows is None:
            return None, None

        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = _scope(best)
        current = [r[:] for r in best]
        current_scope = best_scope

        used = set()
        for r in current:
            used.update(_diffs_of_row(r))

        deadline = time.time() + time_budget
        T = 20.0
        cooling = 0.997
        no_improve = 0

        while time.time() < deadline and T > 0.05:
            ri = rng.randint(0, n_rows - 1)
            old_row = current[ri]
            old_diffs = _diffs_of_row(old_row)
            used_without = used - old_diffs

            new_row = None
            for attempt in range(15):
                new_row = _repair_row(used_without, current_scope, rng, seed_val + attempt)
                if new_row is not None:
                    break

            if new_row is None:
                T *= cooling
                continue

            new_diffs = _diffs_of_row(new_row)
            if new_diffs & used_without:
                T *= cooling
                continue

            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    other_max = max(other_max, max(current[i]))
            new_scope = max(other_max, max(new_row))

            delta = new_scope - current_scope
            if delta <= 0 or (T > 0.1 and rng.random() < math.exp(-delta / T)):
                used = used_without | new_diffs
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            T *= cooling

            if no_improve > 200:
                break

        return best, best_scope

    def _element_adjust(rows, time_budget, seed_val=0):
        """Try to reduce scope by adjusting individual elements downward."""
        if rows is None:
            return None, None

        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = _scope(best)
        current = [r[:] for r in best]
        current_scope = best_scope

        used = set()
        for r in current:
            used.update(_diffs_of_row(r))

        deadline = time.time() + time_budget
        steps = 0

        while time.time() < deadline:
            steps += 1
            ri = rng.randint(0, n_rows - 1)
            pi = rng.randint(1, k)

            old_val = current[ri][pi]
            if old_val <= current[ri][pi - 1] + 1:
                continue

            old_diffs = _diffs_of_row(current[ri])
            used_without = used - old_diffs

            new_val = old_val - 1
            found = False
            while new_val > current[ri][pi - 1]:
                bad = False
                new_diffs = set()
                for i in range(k + 1):
                    if i == pi:
                        continue
                    if i < pi:
                        d = new_val - current[ri][i]
                        if d <= 0 or d in used_without:
                            bad = True
                            break
                        new_diffs.add(d)
                    else:
                        d = current[ri][i] - new_val
                        if d <= 0 or d in used_without:
                            bad = True
                            break
                        new_diffs.add(d)

                if not bad:
                    current[ri][pi] = new_val
                    used = used_without | new_diffs
                    current_scope = _scope(current)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best = [r[:] for r in current]
                    found = True
                    break
                new_val -= 1

            if not found and steps % 100 == 0:
                # Occasionally try a random row replacement
                if _replace_row(current, ri, used, current_scope, rng, seed_val + steps):
                    current_scope = _scope(current)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best = [r[:] for r in current]

        return best, best_scope

    # Main logic
    best_rows = None
    best_M = float('inf')

    # Phase 1: Get a baseline
    start_M = total_diffs
    end_M = total_diffs + 50

    for M in range(start_M, end_M):
        if time.time() > deadline - 50:
            break

        found = False
        for seed in range(12):
            if time.time() > deadline - 45:
                break

            if seed % 4 == 0:
                rows = _interleaved_greedy(M, seed)
            elif seed % 4 == 1:
                rows = _sequential_greedy(M, seed, skip_prob=0.0)
            elif seed % 4 == 2:
                rows = _sequential_greedy(M, seed, skip_prob=0.1)
            else:
                rows = _interleaved_greedy(M, seed + 100)

            if rows is not None and _valid(rows):
                current_max = _scope(rows)
                if current_max < best_M:
                    best_M = current_max
                    best_rows = [r[:] for r in rows]
                found = True
                break

        if best_rows is not None and best_M <= M:
            break

    if best_rows is None:
        for M in range(end_M, end_M + 200):
            if time.time() > deadline - 15:
                break
            rows = _sequential_greedy(M, 0, skip_prob=0.0)
            if rows is not None and _valid(rows):
                best_M = _scope(rows)
                best_rows = [r[:] for r in rows]
                break

    if best_rows is None:
        # Emergency: large gaps
        best_rows = []
        base = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                base += 500
                row.append(base)
            best_rows.append(row)
        best_M = _scope(best_rows)

    # Phase 2: Multi-start hill climbing
    time_left = deadline - time.time()
    if time_left > 15:
        # Run hill climb with different seeds
        for hc_seed in [42, 123, 777, 2024]:
            time_left = deadline - time.time()
            if time_left < 10:
                break
            hc_budget = min(time_left - 8, 25)
            if hc_budget > 0:
                improved, improved_M = _hill_climb(best_rows, hc_budget, seed_val=hc_seed)
                if improved is not None and improved_M < best_M:
                    best_M = improved_M
                    best_rows = improved

    # Phase 3: Element adjustment
    time_left = deadline - time.time()
    if time_left > 8:
        ls_budget = min(time_left - 5, 20)
        if ls_budget > 0:
            improved, improved_M = _element_adjust(best_rows, ls_budget, seed_val=999)
            if improved is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved

    # Phase 4: Final hill climb pass
    time_left = deadline - time.time()
    if time_left > 5:
        hc_budget = min(time_left - 2, 10)
        if hc_budget > 0:
            improved, improved_M = _hill_climb(best_rows, hc_budget, seed_val=555)
            if improved is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved

    return best_rows

if __name__ == "__main__":
    rows = construct(7, 5)
    if rows:
        print("Solution found:")
        for r in rows:
            print(r)
        print("Scope:", _scope(rows))
        print("Valid:", _valid(rows))
    else:
        print("No solution found")
