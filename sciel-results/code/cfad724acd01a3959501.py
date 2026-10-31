"""Improved program for difference triangle set (4,7)."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    # Total number of differences per row: k*(k+1)//2
    # For k=7: 28 differences per row
    # We need n_rows * 28 distinct positive differences

    # Helper: compute all positive differences for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: check if a set of rows is valid (all diffs distinct across rows)
    def check_valid(rows):
        all_diffs = set()
        for row in rows:
            d = row_diffs(row)
            if d & all_diffs:
                return False
            all_diffs |= d
        return True

    # Greedy construction with randomization
    def greedy_build(n_rows, k, used_diffs=None, randomize=False):
        if used_diffs is None:
            used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

    # Greedy with random start offsets
    def greedy_randomized(n_rows, k, seed):
        rng = random.Random(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            attempts = 0
            while len(row) < k + 1:
                # Sometimes try a jump to escape local patterns
                if randomize and rng.random() < 0.1:
                    x += rng.randint(2, 5)
                else:
                    x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
                attempts += 1
                if attempts > 10000:
                    break
            if len(row) < k + 1:
                return None
            rows.append(row)
        return rows

    # Local search: try to reduce the max entry by modifying rows
    def local_search(rows, n_rows, k, max_iters=5000):
        best_rows = [list(r) for r in rows]
        best_max = max(max(r) for r in best_rows)
        used_diffs = set()
        for row in best_rows:
            used_diffs |= row_diffs(row)

        for it in range(max_iters):
            # Pick a random row to modify
            ri = random.randint(0, n_rows - 1)
            old_row = list(best_rows[ri])
            old_diffs = row_diffs(old_row)

            # Try to compress the row: pick a random position and try smaller values
            if len(old_row) <= k:
                break

            # Try removing the largest element and rebuilding the tail
            # Strategy: pick a random index to "restart" from
            restart_idx = random.randint(1, k)  # 1 to k
            prefix = old_row[:restart_idx]
            prefix_diffs = row_diffs(prefix)

            # Check prefix diffs don't conflict with other rows
            other_diffs = set()
            for j, row in enumerate(best_rows):
                if j != ri:
                    other_diffs |= row_diffs(row)

            if prefix_diffs & other_diffs:
                continue

            # Rebuild the tail greedily from prefix
            new_row = list(prefix)
            tail_diffs = set()
            for a in prefix:
                for b in prefix:
                    if a != b:
                        tail_diffs.add(abs(a - b))
            # Actually compute prefix diffs properly
            tail_diffs = row_diffs(prefix)

            x = prefix[-1]
            success = True
            while len(new_row) < k + 1:
                x += 1
                new_diffs = {x - a for a in new_row}
                if new_diffs & other_diffs or new_diffs & tail_diffs:
                    continue
                new_row.append(x)
                tail_diffs |= new_diffs

            if len(new_row) < k + 1:
                continue

            new_max = max(new_row)
            if new_max < best_max:
                # Accept the change
                best_rows[ri] = new_row
                best_max = new_max
                used_diffs = set()
                for row in best_rows:
                    used_diffs |= row_diffs(row)

        return best_rows, best_max

    # Simulated annealing to reduce max entry
    def simulated_annealing(rows, n_rows, k, max_time=60):
        best_rows = [list(r) for r in rows]
        best_max = max(max(r) for r in best_rows)
        current_rows = [list(r) for r in best_rows]
        current_max = best_max

        sa_start = time.time()
        temperature = 10.0
        cooling = 0.999

        while time.time() - sa_start < max_time:
            # Pick a random row and a random position to perturb
            ri = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)

            old_val = current_rows[ri][pos]
            # Try a different value
            delta = random.choice([-2, -1, 1, 2, 3, 5])
            new_val = old_val + delta

            if new_val <= 0 or new_val >= current_rows[ri][pos + 1] if pos < k else new_val >= 0:
                # Check bounds
                if pos > 0 and new_val <= current_rows[ri][pos - 1]:
                    continue
                if pos < k and new_val >= current_rows[ri][pos + 1]:
                    continue

            # Temporarily modify
            old_row = list(current_rows[ri])
            current_rows[ri][pos] = new_val
            new_row = current_rows[ri]

            # Check validity
            new_diffs = row_diffs(new_row)
            other_diffs = set()
            valid = True
            for j, row in enumerate(current_rows):
                if j != ri:
                    other_diffs |= row_diffs(row)

            if new_diffs & other_diffs:
                current_rows[ri] = old_row
                continue

            new_max = max(max(r) for r in current_rows)
            energy_change = new_max - current_max

            if energy_change <= 0 or random.random() < pow(2.718281828, -energy_change / max(temperature, 0.01)):
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in current_rows]
            else:
                current_rows[ri] = old_row

            temperature *= cooling
            if temperature < 0.01:
                temperature = 0.01

        return best_rows, best_max

    # Multi-start approach
    def multi_start(n_rows, k, n_starts, time_budget):
        best_rows = None
        best_max = float('inf')

        for s in range(n_starts):
            if time.time() > start_time + time_budget:
                break

            # Different strategies for different starts
            if s == 0:
                rows = greedy_build(n_rows, k)
            elif s == 1:
                rows = greedy_randomized(n_rows, k, seed=42)
                if rows is None:
                    rows = greedy_build(n_rows, k)
            else:
                rows = greedy_randomized(n_rows, k, seed=s * 12345 + 678)
                if rows is None:
                    continue

            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in rows]

            # Local search refinement
            if time.time() > start_time + time_budget - 5:
                break

            refined, refined_max = local_search(rows, n_rows, k, max_iters=2000)
            if refined_max < best_max:
                best_max = refined_max
                best_rows = refined

        return best_rows, best_max

    # Main algorithm
    # Phase 1: Quick greedy baseline
    rows = greedy_build(n_rows, k)
    best_max = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]

    # Phase 2: Local search on the greedy result
    if time.time() < deadline - 30:
        refined, refined_max = local_search(best_rows, n_rows, k, max_iters=3000)
        if refined_max < best_max:
            best_max = refined_max
            best_rows = refined

    # Phase 3: Multi-start with different random seeds
    remaining_time = deadline - time.time() - 10
    if remaining_time > 10:
        n_starts = min(5, int(remaining_time / 5))
        multi_rows, multi_max = multi_start(n_rows, k, n_starts, remaining_time)
        if multi_max < best_max:
            best_max = multi_max
            best_rows = multi_rows

    # Phase 4: Simulated annealing on the best result
    remaining_time = deadline - time.time() - 5
    if remaining_time > 15:
        sa_rows, sa_max = simulated_annealing(best_rows, n_rows, k, max_time=min(remaining_time - 5, 40))
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows

    # Final local polish
    remaining_time = deadline - time.time() - 2
    if remaining_time > 5:
        polished, polished_max = local_search(best_rows, n_rows, k, max_iters=1000)
        if polished_max < best_max:
            best_max = polished_max
            best_rows = polished

    return best_rows
# EVOLVE-BLOCK-END
