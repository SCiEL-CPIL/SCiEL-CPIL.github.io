import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0

    def _scope(rows):
        return max(a for r in rows for a in r[1:])

    def _valid(rows, k):
        seen = set()
        for r in rows:
            if r[0] != 0 or len(r) != k + 1:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d <= 0 or d in seen:
                        return False
                    seen.add(d)
        return True

    def _build_diff_counts(rows):
        counts = Counter()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    counts[r[j] - r[i]] += 1
        return counts

    def _cost(rows):
        counts = _build_diff_counts(rows)
        collisions = sum(c - 1 for c in counts.values() if c > 1)
        return _scope(rows) + 1000 * collisions

    def _greedy_construct(n_rows, k, seed=0):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def _simulated_annealing(rows, n_rows, k, deadline, seed=42):
        best_rows = [list(r) for r in rows]
        best_cost = _cost(best_rows)
        current_rows = [list(r) for r in rows]
        current_cost = _cost(current_rows)

        rng = random.Random(seed)
        temp = 200.0
        cooling = 0.9995
        min_temp = 0.1

        while time.time() < deadline - 3 and temp > min_temp:
            move_type = rng.random()

            if move_type < 0.4:
                # Single position delta move
                ri = rng.randrange(n_rows)
                row = current_rows[ri]
                pos = rng.randrange(1, len(row))
                old_val = row[pos]
                delta = rng.randint(-8, 8)
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                row[pos] = new_val
                new_cost = _cost(current_rows)
                delta_c = new_cost - current_cost
                if delta_c < 0 or rng.random() < math.exp(-delta_c / temp):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    row[pos] = old_val

            elif move_type < 0.7:
                # Swap two rows entirely
                if n_rows < 2:
                    temp *= cooling
                    continue
                ri, rj = rng.sample(range(n_rows), 2)
                current_rows[ri], current_rows[rj] = current_rows[rj], current_rows[ri]
                new_cost = _cost(current_rows)
                delta_c = new_cost - current_cost
                if delta_c < 0 or rng.random() < math.exp(-delta_c / temp):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    current_rows[ri], current_rows[rj] = current_rows[rj], current_rows[ri]

            elif move_type < 0.9:
                # Swap two positions in different rows
                if n_rows < 2:
                    temp *= cooling
                    continue
                ri, rj = rng.sample(range(n_rows), 2)
                pi = rng.randrange(1, len(current_rows[ri]))
                pj = rng.randrange(1, len(current_rows[rj]))
                vi = current_rows[ri][pi]
                vj = current_rows[rj][pj]
                current_rows[ri][pi] = vj
                current_rows[rj][pj] = vi
                new_cost = _cost(current_rows)
                delta_c = new_cost - current_cost
                if delta_c < 0 or rng.random() < math.exp(-delta_c / temp):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    current_rows[ri][pi] = vi
                    current_rows[rj][pj] = vj

            else:
                # Insert a new random row (replacing worst row)
                if n_rows < 1:
                    temp *= cooling
                    continue
                # Find row with largest max value
                worst_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
                new_row = [0]
                x = 0
                rng2 = random.Random(rng.randint(0, 10**9))
                attempts = 0
                while len(new_row) < k + 1 and attempts < 5000:
                    x += 1
                    new_row.append(x)
                    attempts += 1
                # Simple greedy for new row
                new_row = [0]
                x = 0
                while len(new_row) < k + 1:
                    x += 1
                    new_row.append(x)
                # This is just sequential; let SA fix it
                # Actually let's use a proper greedy
                new_row = [0]
                x = 0
                while len(new_row) < k + 1:
                    x += 1
                    new_row.append(x)
                current_rows[worst_idx] = new_row
                new_cost = _cost(current_rows)
                delta_c = new_cost - current_cost
                if delta_c < 0 or rng.random() < math.exp(-delta_c / temp):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_rows = [list(r) for r in current_rows]
                else:
                    current_rows[worst_idx] = [0] + list(range(1, k + 1))

            temp *= cooling

        return best_rows, best_cost

    def _local_polish(rows, n_rows, k, deadline):
        best_rows = [list(r) for r in rows]
        best_cost = _cost(best_rows)
        rng = random.Random(999)

        for _ in range(30000):
            if time.time() >= deadline - 2:
                break
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            for delta in range(-3, 4):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                row[pos] = new_val
                new_cost = _cost(best_rows)
                if new_cost < best_cost:
                    best_cost = new_cost
                    break
                else:
                    row[pos] = old_val
            else:
                row[pos] = old_val

        return best_rows, best_cost

    # Phase 1: Multi-start greedy
    best_rows = None
    best_scope = float('inf')
    for seed in range(80):
        if time.time() >= deadline - 30:
            break
        rows = _greedy_construct(n_rows, k, seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Phase 2: SA with multiple seeds
    for sa_seed in range(6):
        if time.time() >= deadline - 15:
            break
        sa_rows, sa_cost = _simulated_annealing(best_rows, n_rows, k, deadline, seed=sa_seed * 137 + 7)
        sa_scope = _scope(sa_rows)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows

    # Phase 3: Local polish
    if time.time() < deadline - 3:
        best_rows, best_cost = _local_polish(best_rows, n_rows, k, deadline)
        best_scope = _scope(best_rows)

    return best_rows
