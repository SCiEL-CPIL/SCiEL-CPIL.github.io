import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112  # Safety margin for 150s limit (evaluator kills at 150s)

    # Precompute number of differences per row
    n_diffs_per_row = k * (k + 1) // 2

    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_global(rows):
        all_diffs = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def greedy_row(used, temp=0.0):
        """Build one row greedily with optional temperature for exploration."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                if temp > 0 and random.random() < temp and x < 300:
                    continue
                row.append(x)
                used |= new_diffs
        return row, used

    def find_replacement(fixed_used, current_row, max_budget):
        """Try many candidate rows; return the best (lowest max) that fits."""
        best_local = None
        best_local_max = max(current_row) if current_row else float('inf')
        temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
        for temp in temps:
            if time.time() > deadline - 5:
                break
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                trial_row, _ = greedy_row(fixed_used, temp=temp)
                if max(trial_row) >= max_budget:
                    continue
                if max(trial_row) < best_local_max:
                    best_local = trial_row
                    best_local_max = max(trial_row)
        return best_local

    def used_from_rows(rows):
        used = set()
        for r in rows:
            used |= get_row_diffs(r)
        return used

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Phase 1: Initial construction with multiple strategies ----------
    best_rows = None
    best_max = float('inf')

    # Strategy A: Pure greedy
    used = set()
    rows_greedy = []
    for _ in range(n_rows):
        row, used = greedy_row(used, temp=0.0)
        rows_greedy.append(row)
    if is_valid_global(rows_greedy):
        mv = max_val(rows_greedy)
        if mv < best_max:
            best_max = mv
            best_rows = [list(r) for r in rows_greedy]

    # Strategy B: Multi-temperature restarts
    random.seed(42)
    temps_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    attempts = 0
    while time.time() < deadline - 30 and attempts < 150:
        attempts += 1
        temp = temps_list[attempts % len(temps_list)]
        used = set()
        trial_rows = []
        success = True
        for _ in range(n_rows):
            if time.time() > deadline - 30:
                success = False
                break
            row, used = greedy_row(used, temp=temp)
            trial_rows.append(row)
        if not success:
            continue
        if is_valid_global(trial_rows):
            mv = max_val(trial_rows)
            if mv < best_max:
                best_max = mv
                best_rows = [list(r) for r in trial_rows]

    if best_rows is None:
        # Fallback: should not happen, but just in case
        return rows_greedy

    # ---------- Phase 2: Row-swap local search ----------
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for i in range(n_rows):
            if time.time() > deadline - 15:
                break
            other_rows = [r for j, r in enumerate(best_rows) if j != i]
            fixed_used = used_from_rows(other_rows)
            current_row = best_rows[i]
            new_row = find_replacement(fixed_used, current_row, best_max)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                if is_valid_global(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                        improved = True

    # ---------- Phase 3: Pair-swap refinement ----------
    if time.time() < deadline - 20:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        random.shuffle(pairs)
        for (i, j) in pairs:
            if time.time() > deadline - 10:
                break
            other_rows = [r for x, r in enumerate(best_rows) if x != i and x != j]
            fixed_used = used_from_rows(other_rows)
            for temp in [0.0, 0.1, 0.2, 0.3]:
                if time.time() > deadline - 10:
                    break
                used = set(fixed_used)
                r1, used = greedy_row(used, temp=temp)
                r2, used = greedy_row(used, temp=temp)
                new_rows = [r if x not in (i, j) else (r1 if x == i else r2)
                            for x, r in enumerate(best_rows)]
                if is_valid_global(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows

    # ---------- Phase 4: Simulated annealing-style row replacement ----------
    current_rows = [r[:] for r in best_rows]
    current_row_diffs = [get_row_diffs(r) for r in current_rows]
    current_max = best_max
    t = 10.0
    cooling = 0.9998

    while time.time() < deadline - 3:
        if t < 0.05:
            t = 8.0  # Restart annealing

        idx = random.randrange(n_rows)

        # Calculate forbidden diffs
        forbidden = set()
        for i in range(n_rows):
            if i != idx:
                forbidden |= current_row_diffs[i]

        # Generate new row
        new_row, _ = greedy_row(forbidden, temp=random.choice([0.0, 0.1, 0.2, 0.3, 0.4]))
        new_row_diffs = get_row_diffs(new_row)

        if new_row_diffs & forbidden:
            t *= cooling
            continue

        # Calculate new global max
        new_max = 0
        for i in range(n_rows):
            if i == idx:
                m = max(new_row)
            else:
                m = max(current_rows[i])
            if m > new_max:
                new_max = m

        # Decide acceptance
        if new_max < current_max:
            current_rows[idx] = new_row
            current_row_diffs[idx] = new_row_diffs
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        elif new_max == current_max:
            if random.random() < 0.15:
                current_rows[idx] = new_row
                current_row_diffs[idx] = new_row_diffs
        else:
            delta = new_max - current_max
            prob = pow(2.71828, -delta / t)
            if random.random() < prob:
                current_rows[idx] = new_row
                current_row_diffs[idx] = new_row_diffs
                current_max = new_max

        t *= cooling

    # ---------- Phase 5: Final polish with row-swap ----------
    if time.time() < deadline - 2:
        for i in range(n_rows):
            if time.time() > deadline - 1:
                break
            other_rows = [r for j, r in enumerate(best_rows) if j != i]
            fixed_used = used_from_rows(other_rows)
            new_row = find_replacement(fixed_used, best_rows[i], best_max)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                if is_valid_global(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows

    return best_rows
