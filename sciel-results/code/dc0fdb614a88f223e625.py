import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    k_plus_1 = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    lower_bound = total_diffs_needed - 1  # differences are 1..M, so max entry >= total_diffs - 1

    def row_diffs_set(row):
        """Return set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def row_diffs_mask(row):
        """Return bitmask of all positive differences in a row."""
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                mask |= (1 << (row[j] - row[i]))
        return mask

    def greedy_construct(n_rows, k):
        """Fast greedy baseline using bitmask."""
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                conflict = False
                for a in row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    for a in row:
                        used_mask |= (1 << (x - a))
                    row.append(x)
            rows.append(row)
        return rows

    def backtracking_search(n_rows, k, max_scope, deadline, rng_seed=0):
        """
        Backtracking search to build n_rows of k+1 increasing integers starting at 0,
        with all pairwise differences distinct across all rows, and max element <= max_scope.
        Uses bitmask for fast difference checking with forward-checking gap pruning.
        """
        rng = random.Random(rng_seed)
        rows = []
        k_plus_1 = k + 1

        used_mask = 0

        def build_row(pos, current_last, row_so_far, mask, remaining_rows):
            if time.time() > deadline:
                return None

            if pos == k_plus_1:
                # Row complete; try to build remaining rows
                if remaining_rows == 0:
                    return [list(r) for r in rows] + [list(row_so_far)]
                # Recursively build next row
                rows.append(list(row_so_far))
                result = build_full_rows(remaining_rows - 1, mask, deadline)
                if result is not None:
                    rows.pop()
                    return result
                rows.pop()
                return None

            # Forward checking: prune candidates using gap sum identity
            # We need (k_plus_1 - pos) more elements (including current), each at least 1 apart
            # The remaining span must accommodate them
            remaining_count = k_plus_1 - pos
            max_x = max_scope - (remaining_count - 1)
            if max_x <= current_last:
                return None

            # Find valid candidates with bias toward smaller values
            valid_candidates = []
            for x in range(current_last + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if d > max_scope:
                        conflict = True
                        break
                    if (mask >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
                    if len(valid_candidates) >= 25:
                        break

            if not valid_candidates:
                return None

            # Bias toward smaller values but with some randomization
            n_smallest = min(12, len(valid_candidates))
            smallest = valid_candidates[:n_smallest]
            if len(valid_candidates) > n_smallest and rng.random() < 0.3:
                extra = rng.sample(valid_candidates[n_smallest:],
                                   min(6, len(valid_candidates) - n_smallest))
                candidates_to_try = smallest + extra
            else:
                candidates_to_try = smallest
            rng.shuffle(candidates_to_try)

            for x in candidates_to_try:
                # Compute new mask bits
                new_bits = 0
                for a in row_so_far:
                    new_bits |= (1 << (x - a))

                new_mask = mask | new_bits
                row_so_far.append(x)

                result = build_row(pos + 1, x, row_so_far, new_mask, remaining_rows)
                if result is not None:
                    row_so_far.pop()
                    return result

                row_so_far.pop()

            return None

        def build_full_rows(remaining, mask, deadline):
            if remaining == 0:
                return [list(r) for r in rows]
            row_so_far = [0]
            result = build_row(1, 0, row_so_far, mask, remaining)
            return result

        return build_full_rows(n_rows, 0, deadline)

    def generate_row_bitmask(used_mask, k, randomize=False, p_smallest=0.8, window=4, max_scope=200):
        """Generate a row given a used bitmask. Returns (row, new_mask_bits) or (None, None)."""
        row = [0]
        current_mask = 0

        for _ in range(k):
            candidates = []
            x = row[-1] + 1
            forbidden = used_mask | current_mask

            while x < max_scope:
                conflict = False
                for a in row:
                    d = x - a
                    if (forbidden >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1

            if not candidates:
                return None, None

            if randomize:
                if random.random() < p_smallest:
                    next_x = candidates[0]
                else:
                    next_x = random.choice(candidates)
            else:
                next_x = candidates[0]

            row.append(next_x)
            for a in row[:-1]:
                current_mask |= (1 << (next_x - a))

        return row, current_mask

    # Phase 1: Greedy baseline
    rows = greedy_construct(n_rows, k)
    used_mask = 0
    for row in rows:
        used_mask |= row_diffs_mask(row)

    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    # Phase 2: Simulated Annealing with bitmask
    T = 10.0
    cooling = 0.995
    min_T = 0.1

    while time.time() < deadline - 20.0:  # Reserve time for backtracking phase
        r_idx = random.randint(0, n_rows - 1)

        old_row = rows[r_idx]
        old_diffs_mask = row_diffs_mask(old_row)

        # Remove old row's diffs from used_mask
        # Since we only OR'd bits, we need to recompute from other rows
        other_mask = 0
        for i in range(n_rows):
            if i != r_idx:
                other_mask |= row_diffs_mask(rows[i])

        max_scope = current_max + 5  # Allow slight expansion during SA
        new_row, new_diffs_mask = generate_row_bitmask(other_mask, k, randomize=True,
                                                       p_smallest=0.8, window=3,
                                                       max_scope=max_scope)

        if new_row is None:
            continue

        # Verify no conflicts
        verify_mask = other_mask | new_diffs_mask
        # Check all bits are unique (they should be by construction)

        new_max_val = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_max_val)

        delta = new_global_max - current_max

        if delta < 0:
            rows[r_idx] = new_row
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            if delta == 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)

            if random.random() < prob:
                rows[r_idx] = new_row
                current_max = new_global_max

        T *= cooling
        if T < min_T:
            T = min_T

    # Phase 3: Targeted backtracking search for lower scope
    if time.time() < deadline - 5.0:
        target_scope = best_max - 1
        attempt = 0
        while target_scope > lower_bound and time.time() < deadline - 3.0:
            found = False
            for restart in range(30):
                if time.time() > deadline - 3.0:
                    break
                solution = backtracking_search(n_rows, k, target_scope, deadline,
                                               rng_seed=attempt * 12345 + restart)
                if solution is not None:
                    # Verify solution
                    all_diffs = set()
                    valid = True
                    for row in solution:
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                d = row[j] - row[i]
                                if d in all_diffs:
                                    valid = False
                                    break
                                all_diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        new_scope = max(max(r) for r in solution)
                        if new_scope < best_max:
                            best_max = new_scope
                            best_rows = [r[:] for r in solution]
                            found = True
                            break
                attempt += 1
            if found:
                # Try to improve further
                target_scope = best_max - 1
            else:
                # If we couldn't find at this scope, try one less
                if target_scope <= lower_bound + 3:
                    break
                target_scope -= 1

    # Phase 4: Final polish - local search to tighten each row
    if time.time() < deadline - 2.0:
        # Recompute used mask from best_rows
        used_mask = 0
        for row in best_rows:
            used_mask |= row_diffs_mask(row)

        polish_deadline = deadline - 1.0
        for pass_num in range(3):
            if time.time() > polish_deadline:
                break
            for r_idx in range(n_rows):
                if time.time() > polish_deadline:
                    break
                old_row = best_rows[r_idx]
                old_mask = row_diffs_mask(old_row)
                other_mask = used_mask & ~old_mask

                # Try to regenerate with tighter scope
                current_row_max = max(old_row)
                for tight_scope in range(current_row_max, lower_bound, -1):
                    new_row, new_mask = generate_row_bitmask(other_mask, k,
                                                             randomize=False,
                                                             p_smallest=1.0,
                                                             window=1,
                                                             max_scope=tight_scope)
                    if new_row is not None:
                        # Verify
                        verify = other_mask | new_mask
                        # Check no conflicts by construction
                        if max(new_row) < current_row_max:
                            best_rows[r_idx] = new_row
                            used_mask = other_mask | new_mask
                            best_max = max(max(r) for r in best_rows)
                            break

    return best_rows
