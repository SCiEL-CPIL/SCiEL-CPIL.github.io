import time
import random
from collections import Counter

def construct(n_rows, k):
    deadline = time.time() + 142.0

    def max_entry(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    # --- Randomized greedy: pick next element from candidates with some randomness ---
    def randomized_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Collect all valid next values up to a reasonable bound
                candidates = []
                x = row[-1] + 1
                # Search up to some bound; keep expanding if needed
                bound = max(row[-1] * 2 + 50, 200)
                while x <= bound:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used and len(new_diffs) == len(row):
                        candidates.append(x)
                    x += 1
                if not candidates:
                    # Expand search
                    bound *= 2
                    x = row[-1] + 1
                    while x <= bound:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used and len(new_diffs) == len(row):
                            candidates.append(x)
                        x += 1
                if not candidates:
                    # Fallback: just take the first valid
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used and len(new_diffs) == len(row):
                            candidates = [x]
                            break
                        x += 1

                # Randomized selection: bias toward smaller but allow exploration
                # Pick from the first few candidates with some probability, or a random one
                if len(candidates) > 1:
                    # 70% chance pick from top 3, 30% random from all
                    if random.random() < 0.7:
                        top_n = min(3, len(candidates))
                        val = candidates[random.randint(0, top_n - 1)]
                    else:
                        val = random.choice(candidates)
                else:
                    val = candidates[0]

                row.append(val)
                used.update({val - a for a in row[:-1]})
            rows.append(row)
        return rows

    # --- Deterministic greedy (baseline) ---
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used and len(new) == len(row):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # --- Bit-based monotonic local search (from inspiration 1) ---
    def local_search(rows, time_limit=30.0):
        """Try to reduce max_entry via single-entry moves that preserve validity."""
        current = [r[:] for r in rows]
        current_max = max_entry(current)

        # Precompute used diffs as a set for fast lookup
        used_diffs = set()
        for row in current:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])

        positions = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
        start_time = time.time()
        improved = True

        while improved and time.time() - start_time < time_limit:
            improved = False
            random.shuffle(positions)

            for r, i in positions:
                if time.time() - start_time >= time_limit:
                    break
                row = current[r]
                old_val = row[i]
                lower = row[i - 1] + 1
                upper = row[i + 1] - 1 if i < k else old_val

                # Only try to reduce (or keep same) the max
                # If this isn't the max element, upper can stay; if it is, try smaller
                for new_val in range(lower, upper + 1):
                    if new_val == old_val:
                        continue

                    # Compute old diffs involving position i
                    old_diffs = []
                    for j in range(len(row)):
                        if j == i:
                            continue
                        if j < i:
                            old_diffs.append(old_val - row[j])
                        else:
                            old_diffs.append(row[j] - old_val)

                    # Temporarily remove old diffs
                    temp_used = used_diffs - set(old_diffs)

                    # Check new diffs
                    new_diffs = []
                    conflict = False
                    for j in range(len(row)):
                        if j == i:
                            continue
                        if j < i:
                            nd = new_val - row[j]
                        else:
                            nd = row[j] - new_val
                        if nd in temp_used:
                            conflict = True
                            break
                        new_diffs.append(nd)

                    if conflict:
                        continue
                    if len(set(new_diffs)) != len(new_diffs):
                        continue

                    # Compute new max
                    new_row = row[:]
                    new_row[i] = new_val
                    new_max = max(max(row) if ri != r else max(new_row)
                                  for ri, row in enumerate(current))

                    if new_max < current_max:
                        # Accept
                        used_diffs = temp_used | set(new_diffs)
                        current[r][i] = new_val
                        current_max = new_max
                        improved = True
                        break

        return current, current_max

    # --- Simulated Annealing polish ---
    def sa_polish(rows, time_limit=25.0):
        current = [r[:] for r in rows]
        current_max = max_entry(current)

        diff_counts = Counter()
        for row in current:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff_counts[row[j] - row[i]] += 1

        def get_violations():
            return sum(c - 1 for c in diff_counts.values() if c > 1)

        current_violations = get_violations()
        current_cost = current_violations * 100000 + current_max

        T = 30.0
        T_min = 0.01
        cooling = 0.9999

        indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
        start_time = time.time()

        while time.time() - start_time < time_limit:
            if T < T_min:
                break

            r, i = random.choice(indices)
            row = current[r]
            old_val = row[i]
            lower = row[i - 1] + 1
            upper = row[i + 1] - 1 if i < k else row[i] + 30

            if random.random() < 0.4:
                span = max(1, upper - lower + 1)
                new_val = lower + int(random.random() * span * 0.6)
            else:
                new_val = random.randint(lower, upper)

            if new_val == old_val or new_val < lower or (i < k and new_val > upper):
                continue

            removed_diffs = []
            added_diffs = []
            for j in range(len(row)):
                if j == i:
                    continue
                if j < i:
                    removed_diffs.append(old_val - row[j])
                    added_diffs.append(new_val - row[j])
                else:
                    removed_diffs.append(row[j] - old_val)
                    added_diffs.append(row[j] - new_val)

            for d in removed_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in added_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1

            new_violations = get_violations()
            new_max = max(max(row) for row in current)
            new_cost = new_violations * 100000 + new_max

            delta = new_cost - current_cost
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current[r][i] = new_val
                current_violations = new_violations
                current_cost = new_cost
                if new_max < current_max and new_violations == 0:
                    current_max = new_max
            else:
                for d in added_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in removed_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                current[r][i] = old_val

            T *= cooling

        return current, current_max

    # === Main strategy ===
    # Phase 1: Generate multiple diverse starts
    best_rows = greedy_init()
    best_max = max_entry(best_rows)

    # Try several randomized greedy restarts
    n_restarts = 8
    per_restart_time = 12.0  # seconds per restart for local search
    start_phase = time.time()

    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        rows = randomized_greedy()
        cur_max = max_entry(rows)
        # Only proceed if this start is competitive
        if cur_max <= best_max + 20:
            # Run local search on this start
            ls_time = min(per_restart_time, deadline - time.time() - 5.0)
            if ls_time > 2.0:
                refined, refined_max = local_search(rows, time_limit=ls_time)
                if refined_max < best_max and is_valid(refined):
                    best_rows = refined
                    best_max = refined_max

    # Phase 2: Final local search on the best
    remaining = deadline - time.time() - 30.0
    if remaining > 5.0:
        refined, refined_max = local_search(best_rows, time_limit=min(remaining, 40.0))
        if refined_max < best_max and is_valid(refined):
            best_rows = refined
            best_max = refined_max

    # Phase 3: SA polish
    remaining = deadline - time.time() - 5.0
    if remaining > 3.0:
        polished, polished_max = sa_polish(best_rows, time_limit=min(remaining, 25.0))
        if polished_max < best_max and is_valid(polished):
            best_rows = polished
            best_max = polished_max

    # Final validation
    if not is_valid(best_rows):
        # Fallback to greedy
        best_rows = greedy_init()

    return best_rows
