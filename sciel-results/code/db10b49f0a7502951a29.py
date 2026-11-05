import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit

    n = k + 1
    best_rows = None
    best_score = float('inf')

    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid_row(row):
        if len(row) != n:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        diffs = row_diffs(row)
        expected = len(row) * (len(row) - 1) // 2
        return len(diffs) == expected

    def get_valid_next(row, used, cap):
        valids = []
        last = row[-1]
        for x in range(last + 1, cap + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and len(new_diffs) == len(row):
                valids.append(x)
        return valids

    def randomized_greedy(max_scope_hint=None, seed=0):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < n:
                cap = max_scope_hint if max_scope_hint else 200
                if best_score < float('inf'):
                    cap = min(cap, best_score)

                valids = get_valid_next(row, used, cap)
                if not valids:
                    return None

                # Weighted random choice, favoring smaller x
                if len(valids) > 1:
                    consider = valids[:30]
                    weights = [1.0 / (v + 1) for v in consider]
                    total_w = sum(weights)
                    r = rng.uniform(0, total_w)
                    acc = 0
                    chosen = consider[0]
                    for v, w in zip(consider, weights):
                        acc += w
                        if r <= acc:
                            chosen = v
                            break
                else:
                    chosen = valids[0]

                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows

    def local_search(rows, time_budget):
        rows = [list(r) for r in rows]
        t0 = time.time()
        improved = True
        while improved and time.time() - t0 < time_budget:
            improved = False
            order = list(range(n_rows))
            random.shuffle(order)
            for i in order:
                if time.time() - t0 >= time_budget:
                    break
                row = rows[i]
                od = set()
                for j in range(n_rows):
                    if j != i:
                        od |= row_diffs(rows[j])
                for p in range(len(row) - 1, 0, -1):
                    cur = row[p]
                    lo = row[p - 1] + 1
                    hi = cur - 1
                    if hi < lo:
                        continue
                    for v in range(lo, hi + 1):
                        if p < len(row) - 1 and v >= row[p + 1]:
                            continue
                        cand = list(row)
                        cand[p] = v
                        cd = row_diffs(cand)
                        if len(cd) != n * (n - 1) // 2:
                            continue
                        if cd & od:
                            continue
                        rows[i] = cand
                        improved = True
                        break
                    if improved:
                        break
        return rows

    def validate_solution(rows):
        all_diffs = set()
        for r in rows:
            if not is_valid_row(r):
                return False
            d_set = row_diffs(r)
            if all_diffs & d_set:
                return False
            all_diffs.update(d_set)
        return True

    # Main loop: multiple randomized greedy attempts
    attempts = 0
    while time.time() - start_time < time_limit:
        # Try randomized greedy with different seeds
        seed = attempts * 1234567 + 1
        rows = randomized_greedy(seed=seed)
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                # Try local search
                if time.time() - start_time < time_limit - 5:
                    improved_rows = local_search(best_rows, time_budget=time_limit - 5 - (time.time() - start_time))
                    if validate_solution(improved_rows):
                        improved_score = max(max(r) for r in improved_rows)
                        if improved_score < best_score:
                            best_score = improved_score
                            best_rows = [list(r) for r in improved_rows]
                if best_score <= 105:
                    break

        attempts += 1
        # Occasionally try a more aggressive local search by shuffling row order
        if attempts % 10 == 0 and best_rows is not None and time.time() - start_time < time_limit - 10:
            shuffled = list(best_rows)
            random.shuffle(shuffled)
            improved_rows = local_search(shuffled, time_budget=time_limit - 10 - (time.time() - start_time))
            if validate_solution(improved_rows):
                improved_score = max(max(r) for r in improved_rows)
                if improved_score < best_score:
                    best_score = improved_score
                    best_rows = [list(r) for r in improved_rows]

    if best_rows is None:
        # Fallback to deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < n:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
