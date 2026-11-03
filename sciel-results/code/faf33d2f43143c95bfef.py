import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_row_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                forbidden = used | current_row_diffs
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        row.append(x)
                        current_row_diffs |= new_diffs
                        break
                    x += 1
            rows.append(row)
            used |= current_row_diffs
        return rows

    def local_compact(rows, used_diffs, max_iterations=500):
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for p_idx in range(len(row) - 1, 0, -1):
                    old_val = row[p_idx]
                    if old_val <= 1:
                        continue
                    row.pop(p_idx)
                    diffs_to_remove = set()
                    for a in row:
                        d = abs(old_val - a)
                        diffs_to_remove.add(d)
                    used_diffs -= diffs_to_remove
                    found = False
                    for x in range(old_val - 1, 0, -1):
                        if x in row:
                            continue
                        valid = True
                        for a in row:
                            d = abs(x - a)
                            if d == 0 or d in used_diffs:
                                valid = False
                                break
                        if valid:
                            pos = 0
                            while pos < len(row) and row[pos] < x:
                                pos += 1
                            row.insert(pos, x)
                            for a in row:
                                if a != x:
                                    used_diffs.add(abs(x - a))
                            found = True
                            improved = True
                            break
                    if not found:
                        pos = 0
                        while pos < len(row) and row[pos] < old_val:
                            pos += 1
                        row.insert(pos, old_val)
                        for a in row:
                            if a != old_val:
                                used_diffs.add(abs(old_val - a))
        return rows, used_diffs

    # Core: multi-restart randomized greedy + local compaction
    def randomized_greedy(n_rows, k, seed, max_scope_hint=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_row_diffs = set()
            for _ in range(k):
                x = row[-1] + 1
                forbidden = used | current_row_diffs
                # Collect a window of valid candidates
                candidates = []
                window = rng.randint(3, 8)
                while len(candidates) < window:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        candidates.append(x)
                    x += 1
                if not candidates:
                    # Fallback: extend search
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & forbidden):
                            candidates.append(x)
                            break
                        x += 1
                
                # Pick from candidates with bias toward smaller
                if rng.random() < 0.6:
                    next_x = candidates[0]
                else:
                    next_x = rng.choice(candidates)
                
                row.append(next_x)
                for a in row[:-1]:
                    current_row_diffs.add(next_x - a)
            rows.append(row)
            used |= current_row_diffs
        return rows

    # 1. Initial deterministic greedy
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)

    # 2. Local compaction on initial
    used = get_used_diffs(rows)
    rows, used = local_compact(rows, used)
    compacted_max = max(max(r) for r in rows)
    if compacted_max < best_max:
        best_max = compacted_max
        best_rows = [r[:] for r in rows]

    # 3. Multi-restart randomized greedy + compaction
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        seed = restart * 1000 + 42
        r = randomized_greedy(n_rows, k, seed)
        u = get_used_diffs(r)
        r, u = local_compact(r, u)
        m = max(max(row) for row in r)
        if m < best_max:
            best_max = m
            best_rows = [row[:] for row in r]

    # 4. Final compaction pass
    final_rows = [list(r) for r in best_rows]
    used_set = get_used_diffs(final_rows)
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        row_idx = random.randint(0, n_rows - 1)
        row = final_rows[row_idx]
        if len(row) < 2:
            continue
        last = row[-1]
        if last <= 1:
            continue
        removed = row.pop()
        for a in row:
            used_set.discard(removed - a)
        found = False
        for x in range(last - 1, 0, -1):
            if x in row:
                continue
            valid = True
            for a in row:
                d = abs(x - a)
                if d == 0 or d in used_set:
                    valid = False
                    break
            if valid:
                pos = 0
                while pos < len(row) and row[pos] < x:
                    pos += 1
                row.insert(pos, x)
                for a in row:
                    if a != x:
                        used_set.add(abs(x - a))
                found = True
                break
        if not found:
            row.append(removed)
            row.sort()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_set.add(row[j] - row[i])

    final_scope = max(max(r) for r in final_rows)
    if is_valid(final_rows) and final_scope < best_max:
        best_rows = final_rows
        best_max = final_scope

    # 5. Validate and fallback
    if not is_valid(best_rows):
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)

    return best_rows
