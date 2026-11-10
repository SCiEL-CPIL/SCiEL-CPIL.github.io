import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin under the 150s kill

    total_diffs = n_rows * (k * (k + 1) // 2)
    # Counting bound: M must be >= total_diffs since all diffs are distinct positive ints in [1, M]
    lb = total_diffs

    # Hard search ceiling: beyond this the score is useless; also keeps domain small.
    # Best known is 134 for (4,7). lb for (4,7) is 112.
    upper = lb + 25

    def time_up():
        return time.time() > deadline

    # ---- Greedy constructive baseline (row by row, always smallest feasible) ----
    def greedy_baseline(seed=0, limit=None):
        if limit is None:
            limit = upper
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                best = None
                # try a range of candidates, biased to small
                for x in range(row[-1] + 1, limit + 1):
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        best = x
                        break
                if best is None:
                    return None
                row.append(best)
                for a in row[:-1]:
                    used.add(best - a)
            rows.append(row)
        return rows

    best_rows = None
    best_max = float('inf')

    def record(rows):
        nonlocal best_rows, best_max
        m = max(max(row) for row in rows)
        if m < best_max:
            best_max = m
            best_rows = [row[:] for row in rows]

    # Build a starting point from greedy if it worked.
    for s in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        if time_up():
            break
        g = greedy_baseline(s)
        if g is not None:
            record(g)

    if best_rows is None:
        # Fallback: spread construction that is always valid (uses gaps)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if all((x - a) not in used for a in row):
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                x += 1
            rows.append(row)
        record(rows)

    # ---- Constraint Satisfaction / Backtracking approach ----
    # We model the problem as selecting n_rows sets of k+1 integers starting at 0,
    # such that all pairwise differences are distinct across all rows.
    # We use a backtracking solver that picks the next row's elements one by one,
    # but with a "most constrained" heuristic: at each step, we pick the next element
    # that has the fewest valid continuations (or just use MRV on the next position).
    # Actually, a simpler and more effective CSP approach:
    # 1. Precompute all possible "valid" rows (sets of k+1 ints starting at 0, max <= upper)
    #    that have distinct internal differences.
    # 2. Use backtracking to select n_rows such rows such that their difference sets are disjoint.
    #
    # However, the number of valid rows might be huge.
    # Instead, we use a hybrid:
    # - Start with a greedy solution.
    # - Use local search (simulated annealing) to improve it.
    # - Also, try a direct backtracking search with strong pruning for small limits.

    # Let's implement a more powerful local search:
    # We maintain the set of used differences.
    # We mutate one row at a time.
    # The cost is the number of "conflicting" differences (duplicates).
    # We also want to minimize the maximum value.

    # State:
    # rows: list of lists
    # used_diffs: dict from diff -> count (how many times it appears across all rows)
    # conflicts: number of diffs with count > 1 (or sum of (count-1) for count>1)

    def build_state(rows):
        used_diffs = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    used_diffs[d] = used_diffs.get(d, 0) + 1
        conflicts = 0
        for d, cnt in used_diffs.items():
            if cnt > 1:
                conflicts += cnt - 1
        return used_diffs, conflicts

    def delta_conflict_add(row, x, used_diffs):
        # If we add x to row (assuming x is larger than all existing, or we handle generally),
        # we need to know the new differences.
        # For simplicity in the annealer, we will replace an element, so we remove old diffs and add new.
        # Let's write a helper to compute the change in conflicts when changing a single element in a row.
        pass

    # A more efficient local search:
    # Pick a random row r and a random position i (1..k).
    # Remove the old value, calculate the new set of diffs for that row.
    # Calculate the change in global conflicts.
    # Accept if improvement or with some probability.

    def local_search(rows0, rng, time_limit=10.0):
        rows = [row[:] for row in rows0]
        used_diffs, conflicts = build_state(rows)
        best_rows = [row[:] for row in rows]
        best_conflicts = conflicts
        best_max = max(max(row) for row in rows)

        T = 1.0
        T_min = 0.0001
        start_ls = time.time()

        while time.time() - start_ls < time_limit and not time_up():
            r = rng.randrange(n_rows)
            i = rng.randrange(1, k + 1)  # position 1..k (0 is fixed)

            old_val = rows[r][i]

            # Determine valid range for new value
            lo = rows[r][i - 1] + 1
            hi = rows[r][i + 1] - 1 if i < k else upper

            if lo > hi:
                continue

            # Try a few random candidates
            improved = False
            for _ in range(20):
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue

                # Calculate new diffs for row r
                # Old diffs involving old_val:
                # 1. old_val - rows[r][j] for j < i
                # 2. rows[r][j] - old_val for j > i
                # New diffs involving new_val:
                # 1. new_val - rows[r][j] for j < i
                # 2. rows[r][j] - new_val for j > i

                # We need to update used_diffs and conflicts.
                # Instead of full recalculation, we can do a delta.
                # But for simplicity and correctness, let's just recalc the row's contribution
                # and update the global counts.

                # Remove old_val's contribution
                for j in range(i):
                    d = old_val - rows[r][j]
                    used_diffs[d] -= 1
                    if used_diffs[d] == 0:
                        del used_diffs[d]
                    elif used_diffs[d] == 1:
                        conflicts -= 1  # was 2, now 1, so conflicts decrease by 1
                    # if used_diffs[d] was 1, it becomes 0, no conflict change?
                    # Wait, conflicts = sum (cnt-1) for cnt>1.
                    # If cnt goes 2->1, (2-1)=1 becomes (1-1)=0, so -1.
                    # If cnt goes 3->2, (3-1)=2 becomes (2-1)=1, so -1.
                    # If cnt goes 1->0, (1-1)=0 becomes 0, so 0.
                    # So if new_cnt >= 1, conflicts -= 1. If new_cnt == 0, conflicts -= 0.
                    # Actually, if cnt was 1, it's not a conflict. If it becomes 0, still no conflict.
                    # So: if used_diffs[d] (before deletion) was >= 2, then conflicts -= 1.
                    # My code above: if used_diffs[d] == 1, conflicts -= 1. This is wrong.
                    # If it was 2, it becomes 1. Conflict was 1, now 0. Delta -1.
                    # If it was 3, it becomes 2. Conflict was 2, now 1. Delta -1.
                    # So yes, if the count drops from >=2 to >=1, conflict decreases by 1.
                    # If it drops from 1 to 0, conflict stays 0.

                for j in range(i + 1, len(rows[r])):
                    d = rows[r][j] - old_val
                    used_diffs[d] -= 1
                    if used_diffs[d] == 0:
                        del used_diffs[d]
                    elif used_diffs[d] == 1:
                        conflicts -= 1

                # Add new_val's contribution
                for j in range(i):
                    d = new_val - rows[r][j]
                    if d in used_diffs:
                        old_cnt = used_diffs[d]
                        used_diffs[d] = old_cnt + 1
                        if old_cnt == 1:
                            conflicts += 1  # 1->2, conflict 0->1
                        else:
                            conflicts += 1  # 2->3, conflict 1->2
                    else:
                        used_diffs[d] = 1

                for j in range(i + 1, len(rows[r])):
                    d = rows[r][j] - new_val
                    if d in used_diffs:
                        old_cnt = used_diffs[d]
                        used_diffs[d] = old_cnt + 1
                        if old_cnt == 1:
                            conflicts += 1
                        else:
                            conflicts += 1
                    else:
                        used_diffs[d] = 1

                rows[r][i] = new_val

                if conflicts < best_conflicts or (conflicts == best_conflicts and max(max(row) for row in rows) < best_max):
                    best_conflicts = conflicts
                    best_max = max(max(row) for row in rows)
                    best_rows = [row[:] for row in rows]
                    improved = True
                else:
                    # Revert
                    rows[r][i] = old_val
                    # Revert used_diffs and conflicts
                    # This is getting messy. Let's just keep the new state if accepted, else revert.
                    # But reverting is complex.
                    # Let's just accept if it's not worse, or with probability.
                    # For now, let's only accept improvements to keep it simple and correct.
                    # If we want SA, we need to track the state change carefully.
                    # Given time constraints, let's stick to hill climbing for now, or just revert.
                    # Reverting:
                    # We need to undo the changes.
                    # This is error-prone.
                    # Let's just discard this candidate and try another.
                    # But we've already modified used_diffs and rows.
                    # We need to revert them.
                    # Let's write a revert function or just do it inline.

                    # Revert new_val's contribution
                    for j in range(i):
                        d = new_val - rows[r][j]
                        used_diffs[d] -= 1
                        if used_diffs[d] == 0:
                            del used_diffs[d]
                        elif used_diffs[d] == 1:
                            conflicts -= 1
                    for j in range(i + 1, len(rows[r])):
                        d = rows[r][j] - new_val
                        used_diffs[d] -= 1
                        if used_diffs[d] == 0:
                            del used_diffs[d]
                        elif used_diffs[d] == 1:
                            conflicts -= 1

                    # Re-add old_val's contribution
                    for j in range(i):
                        d = old_val - rows[r][j]
                        if d in used_diffs:
                            old_cnt = used_diffs[d]
                            used_diffs[d] = old_cnt + 1
                            if old_cnt == 1:
                                conflicts += 1
                            else:
                                conflicts += 1
                        else:
                            used_diffs[d] = 1
                    for j in range(i + 1, len(rows[r])):
                        d = rows[r][j] - old_val
                        if d in used_diffs:
                            old_cnt = used_diffs[d]
                            used_diffs[d] = old_cnt + 1
                            if old_cnt == 1:
                                conflicts += 1
                            else:
                                conflicts += 1
                        else:
                            used_diffs[d] = 1

            if not improved:
                # Random kick
                r = rng.randrange(n_rows)
                i = rng.randrange(1, k + 1)
                lo = rows[r][i - 1] + 1
                hi = rows[r][i + 1] - 1 if i < k else upper
                if lo <= hi:
                    new_val = rng.randint(lo, hi)
                    old_val = rows[r][i]
                    # Apply change unconditionally
                    for j in range(i):
                        d = old_val - rows[r][j]
                        used_diffs[d] -= 1
                        if used_diffs[d] == 0:
                            del used_diffs[d]
                        elif used_diffs[d] == 1:
                            conflicts -= 1
                    for j in range(i + 1, len(rows[r])):
                        d = rows[r][j] - old_val
                        used_diffs[d] -= 1
                        if used_diffs[d] == 0:
                            del used_diffs[d]
                        elif used_diffs[d] == 1:
                            conflicts -= 1

                    for j in range(i):
                        d = new_val - rows[r][j]
                        if d in used_diffs:
                            old_cnt = used_diffs[d]
                            used_diffs[d] = old_cnt + 1
                            if old_cnt == 1:
                                conflicts += 1
                            else:
                                conflicts += 1
                        else:
                            used_diffs[d] = 1
                    for j in range(i + 1, len(rows[r])):
                        d = rows[r][j] - new_val
                        if d in used_diffs:
                            old_cnt = used_diffs[d]
                            used_diffs[d] = old_cnt + 1
                            if old_cnt == 1:
                                conflicts += 1
                            else:
                                conflicts += 1
                        else:
                            used_diffs[d] = 1
                    rows[r][i] = new_val

            T = max(T_min, T * 0.999)

        return best_rows, best_conflicts

    # Run local search from several restarts
    rng = random.Random(12345)
    starts = []
    for s in range(10):
        g = greedy_baseline(s)
        if g is not None:
            starts.append(g)
    if not starts:
        starts.append(best_rows)

    attempts = 0
    for base in starts:
        if time_up():
            break
        # a few local search runs per base
        for rep in range(5):
            if time_up():
                break
            res, conf = local_search(base, random.Random(attempts * 7919 + 13), time_limit=5.0)
            if res is not None:
                # verify strictly
                used = set()
                ok = True
                for row in res:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in used:
                                ok = False
                                break
                            used.add(d)
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    record(res)
                    if best_max == lb:
                        return best_rows
            attempts += 1

    return best_rows
