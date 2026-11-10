import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 150s limit

    num_elements = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_construct():
        """Greedy construction: for each row, pick the smallest valid next element."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Phase 1: Pure greedy baseline
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = greedy_scope

    # Phase 2: Randomized greedy with restarts
    while time.time() < deadline - 15:
        used = set()
        rows = []
        success = True

        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < num_elements:
                candidates = []
                x = current_max + 1
                max_x = current_max + max(80, k * 8)
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1

                if not candidates:
                    success = False
                    break

                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    # Weight towards smaller values but allow exploration
                    weights = [1.0 / (c - current_max + 1) ** 0.5 for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    next_x = candidates[-1]
                    for c, w in zip(candidates, weights):
                        cumulative += w
                        if r <= cumulative:
                            next_x = c
                            break

                row.append(next_x)
                for a in row[:-1]:
                    used.add(next_x - a)
                current_max = next_x

            if not success:
                break
            rows.append(row)

        if success and is_valid(rows):
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows

    # Phase 3: Local search with multiple move types
    if best_rows is not None and time.time() < deadline - 2:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        current_diffs = compute_diffs(current_rows)

        # Simulated annealing-like local search
        temperature = max(1.0, (current_scope - lower_bound) / 5.0)
        cooling_rate = 0.998
        no_improve_count = 0
        max_no_improve = 2000

        def try_shift(row_idx, current_rows, current_diffs):
            """Try shifting one element in a row."""
            row = current_rows[row_idx]
            if len(row) <= 1:
                return None, current_diffs

            idx = random.randint(1, len(row) - 1)
            delta = random.randint(-20, 20)
            new_val = row[idx] + delta

            if new_val <= row[idx - 1]:
                return None, current_diffs
            if idx < len(row) - 1 and new_val >= row[idx + 1]:
                return None, current_diffs

            # Compute old and new differences involving this element
            old_diffs = set()
            new_diffs = set()
            for j in range(idx):
                old_diffs.add(row[idx] - row[j])
                new_diffs.add(new_val - row[j])
            for j in range(idx + 1, len(row)):
                old_diffs.add(row[j] - row[idx])
                new_diffs.add(row[j] - new_val)

            other_diffs = current_diffs - old_diffs
            if new_diffs & other_diffs:
                return None, current_diffs

            # Valid move
            new_rows = [r[:] for r in current_rows]
            new_rows[row_idx][idx] = new_val
            new_diffs_set = other_diffs | new_diffs
            return new_rows, new_diffs_set

        def try_replace_row(row_idx, current_rows):
            """Try replacing an entire row."""
            other_rows = [r for i, r in enumerate(current_rows) if i != row_idx]
            other_diffs = set()
            for r in other_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        other_diffs.add(r[j] - r[i])

            # Try to build a new row avoiding other_diffs
            new_row = [0]
            current_max = 0
            success = True
            while len(new_row) < num_elements:
                candidates = []
                x = current_max + 1
                max_x = current_max + max(60, k * 6)
                while x <= max_x:
                    new_diffs = {x - a for a in new_row}
                    if not (new_diffs & other_diffs):
                        candidates.append(x)
                    x += 1

                if not candidates:
                    success = False
                    break

                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    weights = [1.0 / (c - current_max + 1) ** 0.5 for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    next_x = candidates[-1]
                    for c, w in zip(candidates, weights):
                        cumulative += w
                        if r <= cumulative:
                            next_x = c
                            break

                new_row.append(next_x)
                for a in new_row[:-1]:
                    other_diffs.add(next_x - a)
                current_max = next_x

            if not success:
                return None, current_diffs

            new_rows = [r[:] for r in current_rows]
            new_rows[row_idx] = new_row
            new_diffs_set = other_diffs | compute_diffs([new_row])
            return new_rows, new_diffs_set

        def try_swap(row_idx, current_rows, current_diffs):
            """Try swapping two elements in the same row (maintaining order)."""
            row = current_rows[row_idx]
            if len(row) <= 2:
                return None, current_diffs

            i, j = sorted(random.sample(range(1, len(row)), 2))
            # Check if swap maintains increasing order
            if row[i] < row[j]:
                # After swap, we need row[j] < row[i], which is already true
                # But we need to check neighbors
                if i > 0 and row[j] <= row[i - 1]:
                    return None, current_diffs
                if j < len(row) - 1 and row[i] >= row[j + 1]:
                    return None, current_diffs

            # Compute old and new differences
            old_diffs = set()
            new_diffs = set()

            # Differences involving element i
            for t in range(len(row)):
                if t != i:
                    old_diffs.add(abs(row[i] - row[t]))
                    new_diffs.add(abs(row[j] - row[t]))

            # Differences involving element j
            for t in range(len(row)):
                if t != j:
                    old_diffs.add(abs(row[j] - row[t]))
                    new_diffs.add(abs(row[i] - row[t]))

            # Remove the difference between i and j (it stays the same)
            old_diffs.discard(abs(row[i] - row[j]))
            new_diffs.discard(abs(row[i] - row[j]))

            other_diffs = current_diffs - old_diffs
            if new_diffs & other_diffs:
                return None, current_diffs

            new_rows = [r[:] for r in current_rows]
            new_rows[row_idx][i], new_rows[row_idx][j] = row[j], row[i]
            new_diffs_set = other_diffs | new_diffs
            return new_rows, new_diffs_set

        while time.time() < deadline - 2 and no_improve_count < max_no_improve:
            row_idx = random.randint(0, len(current_rows) - 1)
            move_type = random.random()

            new_rows = None
            new_diffs_set = None

            if move_type < 0.6:
                # Shift move
                new_rows, new_diffs_set = try_shift(row_idx, current_rows, current_diffs)
            elif move_type < 0.8:
                # Replace row
                new_rows, new_diffs_set = try_replace_row(row_idx, current_rows)
            else:
                # Swap move
                new_rows, new_diffs_set = try_swap(row_idx, current_rows, current_diffs)

            if new_rows is not None:
                new_scope = get_scope(new_rows)

                # Accept or reject based on simulated annealing
                if new_scope <= current_scope or random.random() < math.exp(-(new_scope - current_scope) / temperature):
                    current_rows = new_rows
                    current_scope = new_scope
                    current_diffs = new_diffs_set
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            temperature *= cooling_rate

        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = current_rows

    return best_rows
