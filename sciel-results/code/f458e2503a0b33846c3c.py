"""Difference triangle set via greedy init + simulated annealing refinement."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using greedy initialization followed
    by simulated annealing to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def time_remaining():
        return deadline - time.time()

    def validate(rows):
        """Check if rows form a valid DTS. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            if not row or row[0] != 0:
                return False, float('inf')
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            max_entry = max(max_entry, row[-1])
        return True, max_entry

    def greedy_initial(n_rows, k, rng, perturb=0):
        """
        Greedy initialization with randomization.
        """
        used_diffs = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if time.time() > deadline - 15:
                    return None
                candidates = []
                # Adaptive search window: larger when we have more time
                time_left = deadline - time.time()
                base_limit = 30 + int(20 * (time_left / 145.0))
                search_limit = x + base_limit + (perturb * 12)
                while x < search_limit:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        candidates.append((x, new_diffs))
                    x += 1
                    if time.time() > deadline - 15:
                        return None

                if not candidates:
                    x = search_limit
                    continue

                # Randomize among top candidates for diversity
                if len(candidates) > 1 and rng.random() < 0.5:
                    top = min(4, len(candidates))
                    chosen_idx = rng.randint(0, top - 1)
                    chosen_val, chosen_diffs = candidates[chosen_idx]
                else:
                    chosen_val, chosen_diffs = candidates[0]

                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                x = chosen_val + 1

            rows.append(row)
        return rows

    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def row_diffs(row, ci):
        """Return set of differences involving position ci in row."""
        old_diffs = set()
        for i in range(ci):
            old_diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - row[ci])
        return old_diffs

    def try_move_element(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs) or (False, None).
        """
        row = rows[ri]
        if ci > 0 and new_val <= row[ci - 1]:
            return False, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None

        old_diffs = row_diffs(row, ci)
        temp_used = used_diffs - old_diffs

        new_diffs = set()
        valid = True
        for i in range(ci):
            d = new_val - row[i]
            if d <= 0 or d in temp_used:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            for j in range(ci + 1, len(row)):
                d = row[j] - new_val
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)

        if not valid:
            return False, None

        new_used = temp_used | new_diffs
        return True, new_used

    def simulated_annealing(rows, n_rows, k, max_iters=30000):
        """
        Simulated annealing to reduce the maximum entry.
        Moves: (1) reduce a random element, (2) targeted reduction of max element,
        (3) perturb an element.
        Accept worsening moves with probability exp(-delta/T).
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)
        best_rows = [r[:] for r in rows]

        T = 80.0
        T_min = 1.0
        cooling = 0.9992  # Faster cooling

        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break

            current_score = max(r[-1] for r in rows)

            # Choose a random move type with weighted probabilities
            # Bias toward reduction moves
            move_type = rng.randint(0, 4)

            if move_type in (0, 1, 2):
                # Try to reduce a random element
                ri = rng.randint(0, n_rows - 1)
                ci = rng.randint(1, len(rows[ri]) - 1)
                current_val = rows[ri][ci]
                lower = rows[ri][ci - 1] + 1
                if ci < len(rows[ri]) - 1:
                    upper = rows[ri][ci + 1] - 1
                else:
                    upper = current_val - 1

                if lower > upper:
                    continue

                new_val = rng.randint(lower, upper)
                if new_val == current_val:
                    continue

                success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                if not success:
                    continue

                new_score = max(r[-1] for r in rows)
                delta = new_score - current_score

                if delta <= 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
                    rows[ri][ci] = new_val
                    used_diffs = new_used
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [r[:] for r in rows]

            elif move_type == 3:
                # Targeted: try to reduce the element contributing to the max
                # Find which row has the max last element
                max_row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
                ri = max_row_idx
                # Try reducing the last element or a random other element in that row
                if rng.random() < 0.7:
                    ci = len(rows[ri]) - 1
                else:
                    ci = rng.randint(1, len(rows[ri]) - 1)
                
                current_val = rows[ri][ci]
                lower = rows[ri][ci - 1] + 1
                if ci < len(rows[ri]) - 1:
                    upper = rows[ri][ci + 1] - 1
                else:
                    upper = current_val - 1

                if lower > upper:
                    continue

                # Try multiple candidate values, prefer smaller ones
                num_attempts = min(5, upper - lower + 1)
                for _ in range(num_attempts):
                    new_val = rng.randint(lower, upper)
                    if new_val == current_val:
                        continue

                    success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                    if not success:
                        continue

                    new_score = max(r[-1] for r in rows)
                    delta = new_score - current_score

                    if delta <= 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
                        rows[ri][ci] = new_val
                        used_diffs = new_used
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [r[:] for r in rows]
                        break

            else:
                # Perturb: try a small change (up or down) to escape local optima
                ri = rng.randint(0, n_rows - 1)
                ci = rng.randint(1, len(rows[ri]) - 1)
                current_val = rows[ri][ci]
                # Try a few nearby values
                for offset in [1, -1, 2, -2, 3, -3]:
                    new_val = current_val + offset
                    if ci > 0 and new_val <= rows[ri][ci - 1]:
                        continue
                    if ci < len(rows[ri]) - 1 and new_val >= rows[ri][ci + 1]:
                        continue

                    success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                    if not success:
                        continue

                    new_score = max(r[-1] for r in rows)
                    delta = new_score - current_score

                    if delta <= 0 or (T > T_min and rng.random() < math.exp(-delta / T)):
                        rows[ri][ci] = new_val
                        used_diffs = new_used
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [r[:] for r in rows]
                        break

            T = max(T_min, T * cooling)

        return best_rows, best_score

    # Main search loop
    best_result = None
    best_score = float('inf')
    rng = random.Random(42)

    # Deterministic greedy first
    det_rows = greedy_initial(n_rows, k, rng, perturb=0)
    if det_rows:
        valid, max_entry = validate(det_rows)
        if valid:
            best_result = det_rows
            best_score = max_entry
            improved, improved_score = simulated_annealing(det_rows, n_rows, k)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved

    # Randomized restarts with SA
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break

        perturb = rng.randint(0, 8)
        rows = greedy_initial(n_rows, k, rng, perturb=perturb)
        if not rows:
            continue

        valid, max_entry = validate(rows)
        if not valid:
            continue

        if max_entry < best_score:
            best_score = max_entry
            best_result = rows
            # Use more iterations for promising solutions
            iters = 40000 if max_entry < best_score - 2 else 30000
            improved, improved_score = simulated_annealing(rows, n_rows, k, max_iters=iters)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved
        else:
            # Even if not better than best, SA might find a better path
            # Only do SA if we have time and the solution is close
            if time.time() < deadline - 35 and max_entry <= best_score + 8:
                improved, improved_score = simulated_annealing(rows, n_rows, k, max_iters=20000)
                if improved_score < best_score:
                    best_score = improved_score
                    best_result = improved

    # Final polish: if we have time left, do one more SA pass on the best result
    if best_result is not None and time.time() < deadline - 25:
        final_improved, final_score = simulated_annealing(best_result, n_rows, k, max_iters=50000)
        if final_score < best_score:
            best_score = final_score
            best_result = final_improved

    if best_result is None:
        # Fallback: simple greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_result
# EVOLVE-BLOCK-END
