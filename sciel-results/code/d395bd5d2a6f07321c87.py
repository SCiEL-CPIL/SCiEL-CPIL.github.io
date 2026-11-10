import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    num_elements = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def row_diffs(row):
        """Return set of positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d > 0:
                    diffs.add(d)
        return diffs

    def all_diffs(rows):
        """Return set of all positive differences across all rows."""
        s = set()
        for row in rows:
            s |= row_diffs(row)
        return s

    def greedy_row_builder(used_set, rng, max_window):
        """
        Builds a row avoiding used_set and internal duplicates.
        """
        row = [0]
        local_set = set()
        current_max = 0

        for _ in range(num_elements - 1):
            candidates = []
            start_val = current_max + 1
            end_val = current_max + max_window

            for x in range(start_val, end_val + 1):
                valid = True
                new_local = set()
                for a in row:
                    d = x - a
                    if d <= 0:
                        continue
                    if d in used_set or d in local_set:
                        valid = False
                        break
                    new_local.add(d)
                if valid:
                    candidates.append((x, new_local))

            if not candidates:
                return None

            # Weighted selection favoring smaller values
            weights = [1.0 / (c[0] - current_max + 1) ** 0.5 for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen = candidates[-1]
            for c, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c
                    break

            next_x, new_local = chosen
            row.append(next_x)
            local_set |= new_local
            current_max = next_x

        return row

    def greedy_construct_all(rng, max_window):
        """Construct all rows greedily."""
        used_set = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row_builder(used_set, rng, max_window)
            if row is None:
                return None
            rows.append(row)
            used_set |= row_diffs(row)
        return rows

    # Phase 1: Randomized Greedy Initialization
    num_initial_attempts = 200
    for i in range(num_initial_attempts):
        if time.time() > deadline - 30:
            break

        rng = random.Random(i * 12345 + 67890)
        max_window = random.choice([30, 40, 50, 60, 80])

        rows = greedy_construct_all(rng, max_window)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows

    if best_rows is None:
        rng = random.Random(0)
        best_rows = greedy_construct_all(rng, 100)
        if best_rows is None:
            used_set = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_set:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_set.add(x - a)
                best_rows.append(row)
            best_scope = get_scope(best_rows)

    # Phase 2: Local Search with Simulated Annealing using set-based diffs
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    current_diff_set = all_diffs(current_rows)

    temperature = max(1.0, (current_scope - lower_bound) / 5.0)
    cooling_rate = 0.999
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline - 2 and no_improve_count < max_no_improve:
        row_idx = random.randint(0, len(current_rows) - 1)
        row = current_rows[row_idx]

        move_type = random.random()

        if move_type < 0.7:
            # Move 1: Shift one element
            if len(row) <= 1:
                continue
            idx = random.randint(1, len(row) - 1)
            delta = random.randint(-20, 20)
            new_val = row[idx] + delta

            if new_val <= row[idx - 1]:
                continue
            if idx < len(row) - 1 and new_val >= row[idx + 1]:
                continue

            # Compute old and new differences for this element
            old_diffs = set()
            new_diffs = set()
            for j in range(idx):
                d_old = row[idx] - row[j]
                d_new = new_val - row[j]
                if d_old > 0:
                    old_diffs.add(d_old)
                if d_new > 0:
                    new_diffs.add(d_new)
            for j in range(idx + 1, len(row)):
                d_old = row[j] - row[idx]
                d_new = row[j] - new_val
                if d_old > 0:
                    old_diffs.add(d_old)
                if d_new > 0:
                    new_diffs.add(d_new)

            # Check for conflicts with other rows' differences
            other_diffs = current_diff_set - old_diffs
            if new_diffs & other_diffs:
                continue

            # Check internal consistency: new_diffs should have exactly len(row)-1 unique values
            if len(new_diffs) != len(row) - 1:
                continue

            # Apply move
            old_val = current_rows[row_idx][idx]
            current_rows[row_idx][idx] = new_val
            new_global_set = other_diffs | new_diffs
            new_scope = max(max(r) for r in current_rows)

            if new_scope <= current_scope or random.random() < math.exp(-(new_scope - current_scope) / temperature):
                current_scope = new_scope
                current_diff_set = new_global_set
                no_improve_count = 0
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                    if best_scope <= lower_bound:
                        return best_rows
            else:
                current_rows[row_idx][idx] = old_val
                no_improve_count += 1
        else:
            # Move 2: Rebuild the row
            row_diff_set = row_diffs(row)
            other_diffs = current_diff_set - row_diff_set

            rng = random.Random()
            new_row = greedy_row_builder(other_diffs, rng, 60)

            if new_row is not None:
                new_row_diffs = row_diffs(new_row)
                new_global_set = other_diffs | new_row_diffs
                old_row = current_rows[row_idx]
                current_rows[row_idx] = new_row
                new_scope = max(max(r) for r in current_rows)

                if new_scope <= current_scope or random.random() < math.exp(-(new_scope - current_scope) / temperature):
                    current_scope = new_scope
                    current_diff_set = new_global_set
                    no_improve_count = 0
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= lower_bound:
                            return best_rows
                else:
                    current_rows[row_idx] = old_row
                    no_improve_count += 1
            else:
                no_improve_count += 1

        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1

    # Phase 3: Final deterministic polish - try all small shifts on the best solution
    polish_deadline = deadline - 1
    best_rows = [r[:] for r in best_rows]
    best_scope = get_scope(best_rows)
    best_diff_set = all_diffs(best_rows)

    # Try single-element shifts with delta in [-5, 5]
    for _ in range(3):  # Multiple passes
        if time.time() > polish_deadline:
            break
        improved = False
        for row_idx in range(len(best_rows)):
            if time.time() > polish_deadline:
                break
            row = best_rows[row_idx]
            for idx in range(1, len(row)):
                for delta in range(-5, 6):
                    new_val = row[idx] + delta
                    if new_val <= row[idx - 1]:
                        continue
                    if idx < len(row) - 1 and new_val >= row[idx + 1]:
                        continue

                    old_diffs = set()
                    new_diffs = set()
                    for j in range(idx):
                        d_old = row[idx] - row[j]
                        d_new = new_val - row[j]
                        if d_old > 0:
                            old_diffs.add(d_old)
                        if d_new > 0:
                            new_diffs.add(d_new)
                    for j in range(idx + 1, len(row)):
                        d_old = row[j] - row[idx]
                        d_new = row[j] - new_val
                        if d_old > 0:
                            old_diffs.add(d_old)
                        if d_new > 0:
                            new_diffs.add(d_new)

                    other_diffs = best_diff_set - old_diffs
                    if new_diffs & other_diffs:
                        continue
                    if len(new_diffs) != len(row) - 1:
                        continue

                    new_scope = max(max(best_rows[row_idx][idx] + delta if i == idx else best_rows[row_idx][i] for i in range(len(best_rows[row_idx]))), max(max(r) for r in best_rows if r is not best_rows[row_idx]))
                    if new_scope < best_scope:
                        best_rows[row_idx][idx] = new_val
                        best_diff_set = other_diffs | new_diffs
                        best_scope = new_scope
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_rows
