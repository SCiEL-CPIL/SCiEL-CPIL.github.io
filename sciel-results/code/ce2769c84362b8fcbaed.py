"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound on max entry
    lower_bound = total_diffs

    best_rows = None
    best_max = float('inf')

    # Helper to compute max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)

    # Helper to check validity and return set of used diffs
    def check_validity(rows):
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return False, used_diffs
                    used_diffs.add(d)
        return True, used_diffs

    # Helper to compute cost for SA
    def compute_cost(rows, max_limit):
        used_diffs = {}
        max_entry = 0
        for row in rows:
            if row:
                m = max(row)
                if m > max_entry:
                    max_entry = m
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        used_diffs[d] += 1
                    else:
                        used_diffs[d] = 1

        num_conflicts = 0
        for count in used_diffs.values():
            if count > 1:
                num_conflicts += count - 1

        conflict_penalty = 10000 * num_conflicts

        limit_penalty = 0
        if max_entry > max_limit:
            limit_penalty = 1000 * (max_entry - max_limit)

        return max_entry + conflict_penalty + limit_penalty

    def generate_random_rows(max_val):
        rows = []
        for _ in range(n_rows):
            if max_val < k:
                row = list(range(k + 1))
            else:
                vals = random.sample(range(1, max_val + 1), k)
                vals.sort()
                row = [0] + vals
            rows.append(row)
        return rows

    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Backtracking search with forward checking
    def backtracking_search(seed, max_limit):
        """
        Build rows element by element using backtracking with forward checking.
        We fill positions (row, col) in row-major order.
        Returns True if a valid solution with max < best_max is found.
        """
        rng = random.Random(seed)

        rows = [[0] + [None] * k for _ in range(n_rows)]
        used_diffs = set()

        def solve(r, c):
            nonlocal used_diffs, rows, best_max, best_rows

            if time.time() > deadline:
                return False

            # If we've filled all rows, we're done
            if r == n_rows:
                current_max = max(max(row) for row in rows)
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [row[:] for row in rows]
                return True

            # If c == 0, this position is already 0, move to next
            if c == 0:
                return solve(r, 1)

            # Determine the minimum possible value for this position
            min_val = rows[r][c - 1] + 1

            # Determine the maximum possible value
            if max_limit is not None:
                max_val = min(max_limit, best_max - 1)
            else:
                max_val = best_max - 1 if best_max != float('inf') else 10000

            if min_val > max_val:
                return False

            # Generate candidates
            width = 5
            candidates = []
            for x in range(min_val, max_val + 1):
                valid = True
                new_diffs = set()
                for prev in rows[r][:c]:
                    d = x - prev
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)

                if valid:
                    candidates.append((x, new_diffs))
                    if len(candidates) >= width:
                        break

            if not candidates:
                return False

            # Randomize the order of candidates
            rng.shuffle(candidates)

            for x, new_diffs in candidates:
                rows[r][c] = x
                used_diffs.update(new_diffs)

                if c < k:
                    if solve(r, c + 1):
                        return True
                else:
                    if solve(r + 1, 0):
                        return True

                used_diffs.difference_update(new_diffs)
                rows[r][c] = None

            return False

        success = solve(0, 0)
        return success

    # SA refinement function
    def sa_refine(initial_rows, max_iter=50000):
        """
        Refine a set of rows using simulated annealing.
        Returns the best rows found.
        """
        nonlocal best_max, best_rows

        current_rows = [row[:] for row in initial_rows]

        # Initial validity check
        valid, _ = check_validity(current_rows)
        if not valid:
            # If invalid, we need to fix it. For now, just use SA with cost function
            pass

        T_start = 50.0
        T_end = 0.1
        cooling_rate = 0.9995

        current_cost = compute_cost(current_rows, best_max - 1)

        best_in_sa = current_rows[:]
        best_cost_in_sa = current_cost

        for i in range(max_iter):
            if time.time() > deadline:
                break

            # Generate a neighbor
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)

            min_val = current_rows[r_idx][c_idx - 1] + 1
            max_val = best_max - 1

            if c_idx < k:
                max_val = min(max_val, current_rows[r_idx][c_idx + 1] - 1)

            if min_val > max_val:
                continue

            new_val = random.randint(min_val, max_val)

            if new_val == current_rows[r_idx][c_idx]:
                continue

            old_val = current_rows[r_idx][c_idx]
            current_rows[r_idx][c_idx] = new_val

            new_cost = compute_cost(current_rows, best_max - 1)
            delta_cost = new_cost - current_cost

            T = T_start * (cooling_rate ** i)

            if delta_cost < 0 or (T > 0 and random.random() < math.exp(-delta_cost / T)):
                current_cost = new_cost

                # Check if we found a valid solution better than best
                if new_cost < best_max:
                    valid, _ = check_validity(current_rows)
                    if valid:
                        curr_max = get_max_entry(current_rows)
                        if curr_max < best_max:
                            best_max = curr_max
                            best_rows = [row[:] for row in current_rows]
                            best_in_sa = [row[:] for row in current_rows]
                            best_cost_in_sa = new_cost
            else:
                current_rows[r_idx][c_idx] = old_val

        return best_in_sa

    # Initial baseline
    baseline_rows = pure_greedy()
    best_rows = [row[:] for row in baseline_rows]
    best_max = get_max_entry(best_rows)

    # Try backtracking search first to get a better initial solution
    # We try to find a solution with max < best_max
    current_seed = 0
    num_bt_attempts = 20

    for _ in range(num_bt_attempts):
        if time.time() > deadline:
            break

        current_seed += 1
        # Try backtracking with a limit slightly below current best
        # If we can't find one, we might need to relax, but for now we try to improve
        backtracking_search(current_seed, best_max - 1)

        # If best_max improved, we can continue with tighter bounds
        # If not, we might break early to save time for SA
        if best_max <= lower_bound:
            break

    # Now use SA to refine the best solution found
    # We run multiple SA rounds with different starting points
    num_sa_rounds = 10

    for sa_round in range(num_sa_rounds):
        if time.time() > deadline:
            break

        # Start from the current best
        if sa_round == 0:
            start_rows = [row[:] for row in best_rows]
        else:
            # Start from a perturbed version of the best
            start_rows = [row[:] for row in best_rows]
            # Perturb a few random entries
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, k)
                min_val = start_rows[r_idx][c_idx - 1] + 1
                max_val = min(best_max - 1, start_rows[r_idx][c_idx + 1] - 1 if c_idx < k else best_max - 1)
                if min_val <= max_val:
                    start_rows[r_idx][c_idx] = random.randint(min_val, max_val)

        # Run SA
        sa_refine(start_rows, max_iter=20000)

        # If we hit the lower bound, we can stop
        if best_max <= lower_bound:
            break

    # Final check: ensure the best_rows are valid
    # If not, fall back to greedy (which is always valid)
    valid, _ = check_validity(best_rows)
    if not valid:
        best_rows = pure_greedy()
        best_max = get_max_entry(best_rows)

    return best_rows
# EVOLVE-BLOCK-END
