import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 148  # Safety margin for 150s limit

    # Total number of distinct differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Lower bound for scope M
    # We need total_diffs_needed distinct positive integers <= M
    # So M >= total_diffs_needed
    M_lb = total_diffs_needed

    # Greedy construction to get an initial upper bound
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)

    # If the greedy solution is already at the lower bound, return it
    if best_scope == M_lb:
        return best_rows

    # Strategy: Conflict-Directed Repair (CDR)
    # 1. Initialize rows with a "tight" packing (e.g., small gaps) to start with a small scope.
    #    However, simple tight packing creates many conflicts.
    #    Instead, we initialize with the greedy solution but try to compress it.
    #    Or, better: Start with a random valid solution (if we can find one quickly) or the greedy one.
    #    Then, repeatedly pick the largest element in the largest row, and try to reduce it
    #    to the smallest possible value that doesn't create new conflicts, while maintaining
    #    the strictly increasing order and the distinct difference property.

    # To make CDR effective, we need a representation that allows fast conflict checking.
    # used_diffs is a set.
    # For a given row, we store the list of values.

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_solution(rows):
        used_diffs = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return False
                    used_diffs.add(d)
        return True

    # The CDR heuristic:
    # While time allows:
    #   1. Identify the row with the largest last element (or just the largest element overall).
    #   2. Identify the largest element in that row (which is the last one, since it's increasing).
    #   3. Try to decrease this element to a smaller value v.
    #      - v must be > previous element in the row.
    #      - v must be < current value (to improve scope).
    #      - The set of differences created by v with the rest of the row must be disjoint from
    #        the global used_diffs set (excluding the differences that v currently creates).
    #      - The new differences must also be disjoint from each other (they are, since v is unique in row).
    #   4. If a valid v is found, update the row and the global used_diffs set.
    #   5. If no improvement is found in this row, try the next largest row.
    #   6. If no row can be improved, we might need to "shake" the solution.
    #      How to shake? Randomly pick a row and a position, and try to increase it slightly,
    #      or swap elements? No, elements must be increasing.
    #      Maybe randomize the initial greedy construction or use a different greedy order?
    #      Or, if stuck, restart with a new random seed for a different greedy variant?
    #      Given the time limit, we can run multiple "phases" of CDR with different initializations.

    # Let's implement a more robust CDR.
    # We'll maintain:
    # rows: list of lists
    # used_diffs: set of all differences

    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    # Phase 1: Get a valid starting point.
    # We use the greedy solution.
    current_rows = [row[:] for row in best_rows]
    current_used_diffs = compute_used_diffs(current_rows)
    current_scope = max(max(r) for r in current_rows)

    # Phase 2: CDR Loop
    # We try to reduce the scope by lowering the largest elements.
    # To avoid getting stuck, we can introduce some randomness in the order of trying values
    # or the order of rows.

    # Optimization: Precompute row differences?
    # When we change row[r][c], only the differences involving row[r][c] change.
    # Specifically, for row r, the differences are {row[r][j] - row[r][i] | i < j}.
    # If we change row[r][c] from old_val to new_val:
    # - Remove old differences: {old_val - row[r][i] for i < c} and {row[r][j] - old_val for j > c}
    # - Add new differences: {new_val - row[r][i] for i < c} and {row[r][j] - new_val for j > c}
    # Note: Since the row is increasing, row[r][j] > old_val for j > c is false if c is the last element.
    # In our CDR, we always target the LAST element of a row (index k).
    # So j > c is empty.
    # Thus, we only need to handle differences with i < c.

    # Let's refine the CDR step for the last element of a row.
    # Target: row[r][k]
    # Old value: old_val
    # Differences to remove: {old_val - row[r][i] for i in range(k)}
    # New value: new_val (where row[r][k-1] < new_val < old_val)
    # Differences to add: {new_val - row[r][i] for i in range(k)}
    # Check:
    # 1. New differences must not be in (current_used_diffs - removed_diffs)
    # 2. New differences must not conflict with each other (they don't, as they are distinct)

    def try_reduce_last_element(r, current_rows, current_used_diffs, deadline):
        row = current_rows[r]
        old_val = row[k]
        prev_val = row[k-1]

        # Differences to remove
        removed_diffs = set()
        for i in range(k):
            d = old_val - row[i]
            removed_diffs.add(d)

        # Candidate new values: from prev_val + 1 to old_val - 1
        # We want the smallest possible new_val to maximize reduction?
        # Or just any new_val < old_val?
        # Smaller new_val is better for scope.
        # So we iterate from prev_val + 1 upwards.

        # To make it faster, we can pre-check:
        # The new differences are {new_val - row[i] for i in range(k)}.
        # Let D_new = {new_val - row[i] for i in range(k)}
        # We need D_new.isdisjoint(current_used_diffs - removed_diffs)

        # We can iterate new_val from prev_val + 1 to old_val - 1.
        # For each new_val, compute D_new and check disjointness.

        # Optimization:
        # If we find a valid new_val, we can stop (since we're going from small to large,
        # the first valid one is the smallest, which is best).

        # However, checking disjointness for every new_val might be slow if k is large.
        # k is 7, so range(k) is 7. D_new has 7 elements.
        # current_used_diffs has size ~ n_rows * 28 = 112.
        # Set operations are fast.

        for new_val in range(prev_val + 1, old_val):
            # Check if new_val creates conflicts
            # We need to check if any (new_val - row[i]) is in (current_used_diffs - removed_diffs)
            # Let's create the set of "forbidden" diffs for this step
            # forbidden = current_used_diffs - removed_diffs
            # But we don't want to create a new set every time.
            # We can check: for each i, is (new_val - row[i]) in current_used_diffs?
            # If yes, is it in removed_diffs? If no, conflict.

            valid = True
            for i in range(k):
                d = new_val - row[i]
                if d in current_used_diffs:
                    # Check if it's being removed
                    if d not in removed_diffs:
                        valid = False
                        break
            if not valid:
                continue

            # Also, we must ensure that the new differences themselves are distinct?
            # They are, because row[i] are distinct.

            # Also, we must ensure that the new differences don't conflict with each other?
            # No, they are just a set.

            # If valid, commit
            # Update current_used_diffs
            # Remove removed_diffs
            # Add new_diffs

            # But wait, we are in a "what-if" scenario.
            # We need to actually update the state if we succeed.
            # Since we are iterating, we can just return the new_val and let the caller update.
            # Or we can update in place and return True.

            # Let's update in place.
            new_diffs = set()
            for i in range(k):
                d = new_val - row[i]
                new_diffs.add(d)

            # Remove old
            current_used_diffs.difference_update(removed_diffs)
            # Add new
            current_used_diffs.update(new_diffs)

            # Update row
            row[k] = new_val

            return True

        return False

    # Main CDR Loop
    # We repeat until no improvement is found or time runs out.
    # To escape local minima, we can randomize the order of rows we try.
    # Also, if we get stuck (no row can be reduced), we can try to "perturb" the solution.
    # Perturbation: Pick a row, pick a random element (not the last one), increase it by 1.
    # This increases the scope, but might allow subsequent reductions to go lower.
    # However, increasing an element creates new differences, which might block reductions.
    # A better perturbation: Swap the order of construction?
    # Or, restart the CDR with a different initial greedy solution?

    # Let's try multiple initializations.
    # We'll run CDR for a fixed time, then restart with a new greedy variant.

    def run_cdr(initial_rows, time_limit):
        rows = [row[:] for row in initial_rows]
        used_diffs = compute_used_diffs(rows)
        scope = max(max(r) for r in rows)
        best_scope = scope
        best_rows = [row[:] for row in rows]

        # Order of rows to try
        # We try to reduce the row with the largest last element first.
        # But to be random, we can shuffle the row indices.

        while time.time() < time_limit:
            # Find the row with the largest last element
            # Or just iterate through rows in a random order
            row_indices = list(range(n_rows))
            random.shuffle(row_indices)

            improved = False
            for r in row_indices:
                if time.time() > time_limit:
                    break

                # Try to reduce the last element of row r
                if try_reduce_last_element(r, rows, used_diffs, time_limit):
                    improved = True
                    new_scope = max(max(row) for row in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [row[:] for row in rows]

            if not improved:
                # No improvement in this pass.
                # Try a perturbation?
                # Or break and restart with a new initialization?
                # Since we are in a single CDR run, let's break and let the caller restart.
                # But we might have time for more passes if we shuffle differently?
                # If no row could be reduced, shuffling won't help unless we change the state.
                # So we break.
                break

        return best_rows, best_scope

    # We will run multiple CDR phases with different initializations.
    # Initialization 1: Standard Greedy
    # Initialization 2: Greedy with randomized tie-breaking?
    # Initialization 3: Greedy starting with larger gaps?

    # Let's define a flexible greedy constructor.
    def flexible_greedy(seed):
        random.seed(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Pick the next value
                # Standard greedy picks the smallest valid x > current_max
                # Here, we can pick from a range [current_max + 1, current_max + gap_limit]
                # But we want to keep it small.
                # Let's stick to standard greedy but maybe randomize the order of rows?
                # The rows are independent in greedy, so order doesn't matter for validity,
                # but it matters for the "used_diffs" pool?
                # No, used_diffs is global.
                # So the order of rows matters for the resulting set of differences.
                # Let's randomize the row construction order.

                pass # Placeholder, actual implementation below

        # Actually, let's just use the standard greedy but with different random seeds
        # to shuffle the internal choices if any.
        # The standard greedy is deterministic.
        # To get different solutions, we can modify the greedy rule.
        # Rule: Pick x from [current_max + 1, current_max + 10] that is valid,
        # preferring smaller x but with some probability picking a larger one.

        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Try values in a random order within a window?
                # Or just standard greedy.
                x = current_max + 1
                found = False
                while True:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        found = True
                        break
                    x += 1
                    if x > 1000: # Safety
                        break
                if not found:
                    # Fallback
                    break
            rows.append(row)
        return rows

    # Since flexible_greedy is deterministic for a given seed (if we don't randomize the search),
    # we need to actually randomize it.

    def randomized_greedy(seed):
        random.seed(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Generate a list of candidate values
                # Start from current_max + 1, go up to some limit
                candidates = []
                for delta in range(1, 50):
                    x = current_max + delta
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        if len(candidates) > 10:
                            break

                if not candidates:
                    # Should not happen if limit is high enough
                    x = current_max + 1
                    while True:
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            break
                        x += 1
                    candidates = [x]

                # Pick a candidate
                # Prefer smaller, but with some randomness
                # Weight by 1/x?
                # Or just pick the first one with high probability, next with lower, etc.
                # Let's pick the smallest with prob 0.8, next smallest with 0.1, etc.
                # Or simply: with prob 0.5, pick the smallest. Else, pick a random one from the top 3.
                if random.random() < 0.7:
                    x = candidates[0]
                else:
                    top_n = min(3, len(candidates))
                    x = random.choice(candidates[:top_n])

                row.append(x)
                used_diffs.update(x - a for a in row[:-1])
                current_max = x
            rows.append(row)
        return rows

    # Run CDR with multiple initializations
    best_overall_rows = best_rows
    best_overall_scope = best_scope

    # Time budget per phase
    # Total time ~ 148s
    # We can run, say, 10 phases of 10s each?
    # Or fewer phases with longer time?
    # CDR is fast if it makes progress. If it gets stuck, it breaks quickly.
    # So we can run many short phases.

    num_phases = 20
    time_per_phase = 6.0

    for phase in range(num_phases):
        if time.time() > deadline - 2:
            break

        # Generate a new initialization
        # Phase 0 uses the original greedy
        if phase == 0:
            init_rows = best_rows
        else:
            seed = phase * 12345 + 6789
            init_rows = randomized_greedy(seed)

        # Check if init is valid (it should be)
        if not is_valid_solution(init_rows):
            continue

        init_scope = max(max(r) for r in init_rows)
        if init_scope >= best_overall_scope:
            # This initialization is already worse, but CDR might improve it below best_overall
            # So we still run it.
            pass

        # Run CDR
        phase_deadline = time.time() + time_per_phase
        cdr_rows, cdr_scope = run_cdr(init_rows, phase_deadline)

        if cdr_scope < best_overall_scope:
            best_overall_scope = cdr_scope
            best_overall_rows = cdr_rows

    return best_overall_rows

# Test/Validation
if __name__ == "__main__":
    rows = construct(4, 7)
    print(rows)
    print("Scope:", max(max(r) for r in rows))
    # Validate
    used = set()
    valid = True
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in used:
                    valid = False
                used.add(d)
    print("Valid:", valid)
