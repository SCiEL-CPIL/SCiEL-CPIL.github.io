"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs_set(row):
                if d in seen:
                    return False
                seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)

    # ---------- Bin-packing inspired row construction ----------
    # The idea: when building a row, we want to choose entries such that the
    # resulting set of differences is "balanced" - not too many small diffs
    # that could be used by other rows, but also not forced to use huge gaps.
    # We use a scoring function to evaluate partial rows.

    def score_partial_row(row, used_other, current_diffs):
        """Score a partial row: lower is better.
        Penalize:
        - Using very small differences (wastes cheap diffs)
        - Having a large gap to the next needed entry
        - Creating differences that are hard to fit elsewhere
        """
        if not row:
            return 0
        max_entry = row[-1]
        # Number of differences used so far
        n_diffs = len(current_diffs)
        # Expected total diffs for a full row
        total_diffs = (k + 1) * k // 2
        # Penalty for using small diffs (they're more contested)
        small_penalty = sum(1 for d in current_diffs if d <= 20) * 0.5
        # Penalty for the span relative to what's needed
        # A row of k+1 entries needs at least k*(k+1)/2 distinct diffs
        # The minimal span for m diffs is roughly sqrt(2*m)
        min_span_est = int(math.sqrt(2 * total_diffs)) + 1
        span_penalty = max(0, max_entry - min_span_est) * 0.1
        # Bonus for using a good mix
        if n_diffs > 0:
            avg_diff = sum(current_diffs) / n_diffs
            # We want avg_diff to be around min_span_est/2
            mix_penalty = abs(avg_diff - min_span_est / 2) * 0.05
        else:
            mix_penalty = 0
        return small_penalty + span_penalty + mix_penalty

    def build_row_smart(used_other, max_span, num_restarts=5):
        """Build a row using a smarter heuristic that considers difference distribution.
        Returns the best row found, or None if no valid row within max_span.
        """
        best_row = None
        best_score = float('inf')

        for restart in range(num_restarts):
            row = [0]
            current_diffs = set()
            x = 0
            success = True

            while len(row) < k + 1:
                # Find all valid next positions
                candidates = []
                for trial_x in range(x + 1, max_span + 1):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_diffs):
                        # Score this candidate
                        trial_diffs = current_diffs | new
                        trial_row = row + [trial_x]
                        sc = score_partial_row(trial_row, used_other, trial_diffs)
                        candidates.append((sc, trial_x, new))

                if not candidates:
                    success = False
                    break

                # Sort by score and pick from the top few with some randomness
                candidates.sort(key=lambda x: x[0])
                top_n = min(5, len(candidates))
                # Weighted random selection from top candidates
                weights = [1.0 / (i + 1) for i in range(top_n)]
                total_w = sum(weights)
                r = random.random() * total_w
                cum = 0
                chosen = candidates[0]
                for i in range(top_n):
                    cum += weights[i]
                    if r <= cum:
                        chosen = candidates[i]
                        break

                sc, trial_x, new = chosen
                row.append(trial_x)
                current_diffs |= new
                x = trial_x

            if success:
                row_score = score_partial_row(row, used_other, current_diffs)
                if row_score < best_score:
                    best_score = row_score
                    best_row = list(row)

        return best_row

    def try_reduce_row_v2(rows, row_idx, used_other, max_attempts=100):
        """Try to find a row at row_idx with smaller max entry using smart construction."""
        current_row = rows[row_idx]
        current_max = max(current_row)
        best_row = list(current_row)
        best_max = current_max

        for attempt in range(max_attempts):
            if time_left() < 3:
                break

            # Decrease the search space gradually
            target_max = current_max - 1 - (attempt // 10)
            if target_max < k:  # Can't have less than k
                break

            # Try smart construction with decreasing max_span
            row = build_row_smart(used_other, target_max, num_restarts=3)
            if row is not None:
                # Verify it's valid in context
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    if max(row) < best_max:
                        best_row = list(row)
                        best_max = max(row)

        # Also try a few randomized greedy attempts as fallback
        for attempt in range(20):
            if time_left() < 3:
                break
            row, x = [0], 0
            current_row_diffs = set()
            success = True
            jitter = 0.1 if attempt > 5 else 0.0
            while len(row) < k + 1:
                found = False
                for trial_x in range(x + 1, x + 60):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & current_row_diffs):
                        if jitter > 0 and random.random() < jitter and trial_x < 200:
                            continue
                        row.append(trial_x)
                        current_row_diffs |= new
                        found = True
                        break
                if not found:
                    success = False
                    break
                x = row[-1]
            if success and max(row) < best_max:
                test_rows = [list(r) for r in rows]
                test_rows[row_idx] = row
                if is_valid(test_rows):
                    best_row = list(row)
                    best_max = max(row)

        return best_row, best_max

    def local_polish(rows):
        """Reduce scope by locally re-optimizing each row (ordered by scope)."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 8
        pass_count = 0
        while improved and time_left() > 8 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time_left() < 8:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                new_row, new_max = try_reduce_row_v2(rows, i, used_other, max_attempts=60)
                if new_max < max(rows[i]):
                    rows[i] = new_row
                    improved = True
        return rows

    def kick(rows):
        """Perturb the solution to escape local minima."""
        rows = [list(r) for r in rows]
        num_kicks = random.choice([1, 2])
        indices = random.sample(range(n_rows), num_kicks)

        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])

            # Try smart rebuild with high randomness
            success = False
            for attempt in range(15):
                # Use a larger max_span for kicks to allow exploration
                max_span = max(rows[idx]) + 30 + attempt * 10
                row = build_row_smart(used_other, max_span, num_restarts=2)
                if row is not None:
                    rows[idx] = row
                    success = True
                    break

            if not success:
                # Fallback to greedy
                greedy_row, x = [0], 0
                row_diffs = set()
                while len(greedy_row) < k + 1:
                    x += 1
                    new = {x - a for a in greedy_row}
                    if not (new & used_other) and not (new & row_diffs):
                        greedy_row.append(x)
                        row_diffs |= new
                rows[idx] = greedy_row

        return rows

    # Initial polish
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    while time_left() > 25:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        # 2. Kick
        if time_left() < 20:
            break

        kicked_rows = kick(best_rows)

        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)

        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final polish
    if time_left() > 5:
        while time_left() > 3:
            polished = local_polish(best_rows)
            if scope(polished) < best_scope:
                best_rows = polished
                best_scope = scope(polished)
            else:
                break

    # Final validity guarantee
    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
