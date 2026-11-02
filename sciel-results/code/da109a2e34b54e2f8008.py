import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, gap_factor=10):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used_set(rows):
    """Compute the union of all differences across all rows (as a set)."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _compute_all_used_ba(rows, max_diff):
    """Compute the union of all differences across all rows as a bytearray."""
    ba = bytearray(max_diff + 1)
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                if d <= max_diff:
                    ba[d] = 1
    return ba

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _build_static_diffs(row, k, pair_idx):
    """
    For each position pi (1..k), compute the set of differences in the row
    that do NOT involve position pi.
    """
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for (i, j) in pair_idx:
            if i == pi or j == pi:
                continue
            s.add(row[j] - row[i])
        static[pi] = s
    return static

def _pair_indices(k):
    """Precompute (i, j) pairs for a row of length k+1."""
    n = k + 1
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _try_move_fast(current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                   row_maxes, k, max_diff):
    """
    Try to apply a move. Returns (success, new_diffs, cand_row_max, new_global_max).
    """
    row = current_rows[ri]
    if new_val <= row[pi - 1]:
        return False, None, None, None
    if pi < k and new_val >= row[pi + 1]:
        return False, None, None, None

    static_pi = static_diffs[ri][pi]
    old_diffs = row_diffs[ri]

    # Compute the k diffs involving pi and check for internal conflicts
    new_diffs = set(static_pi)
    valid = True
    for i in range(len(row)):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in new_diffs:
            valid = False
            break
        new_diffs.add(d)

    if not valid:
        return False, None, None, None

    # Check for conflicts with other rows via the bytearray
    for d in new_diffs:
        if d in old_diffs:
            continue
        if d <= max_diff and all_used_ba[d]:
            return False, None, None, None

    if pi == k:
        cand_row_max = new_val
    else:
        cand_row_max = row[k]

    old_row_max = row_maxes[ri]
    row_maxes[ri] = cand_row_max
    new_global_max = max(row_maxes)
    row_maxes[ri] = old_row_max

    return True, new_diffs, cand_row_max, new_global_max

def _apply_move_fast(current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                     row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                     pair_idx, max_diff):
    """Apply a valid move to the state."""
    for d in old_diffs:
        if d <= max_diff:
            all_used_ba[d] = 0
    for d in new_diffs:
        if d <= max_diff:
            all_used_ba[d] = 1
    current_rows[ri][pi] = new_val
    row_diffs[ri] = new_diffs
    row_maxes[ri] = cand_row_max
    static_diffs[ri] = _build_static_diffs(current_rows[ri], k, pair_idx)
    return new_global_max if False else max(row_maxes)

def _sa_phase(current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
              current_val, best_rows, best_val, T, T_min, cool, max_no_improve,
              deadline, pair_idx, max_diff):
    """Run simulated annealing phase. Returns updated state."""
    no_improve = 0

    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)

        old_val = current_rows[ri][pi]
        old_diffs = row_diffs[ri]

        # Generate candidates
        candidates = []

        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(60, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)

        # Increasing candidates
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
        max_increase = min(20, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)

        # Random jump
        if random.random() < 0.2:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            if low <= high:
                candidates.append(random.randint(low, high))

        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue

        random.shuffle(candidates)

        moved = False
        for new_val in candidates:
            success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                row_maxes, k, max_diff)

            if not success:
                continue

            delta = new_global_max - current_val

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.25:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True

            if accept:
                current_val = _apply_move_fast(
                    current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                    row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                    pair_idx, max_diff)

                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1

                moved = True
                break

        if not moved:
            no_improve += 1

        T = max(T_min, T * cool)

        # Periodic restart if stuck
        if no_improve > 15000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used_ba = _compute_all_used_ba(current_rows, max_diff)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0

    return current_rows, current_val, all_used_ba, row_diffs, static_diffs, row_maxes, best_rows, best_val

def _polish_phase(current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
                  current_val, best_rows, best_val, deadline, pair_idx, max_diff):
    """Final polishing with only decreasing moves and tight tolerance."""
    T = 1.0
    T_min = 0.05
    cool = 0.99998
    no_improve = 0
    max_no_improve_polish = 15000

    while time.time() < deadline and no_improve < max_no_improve_polish:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)

        old_val = current_rows[ri][pi]
        old_diffs = row_diffs[ri]

        # Only decreasing moves in polish
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(25, old_val - min_allowed)

        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)

        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue

        random.shuffle(candidates)

        moved = False
        for new_val in candidates:
            success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                row_maxes, k, max_diff)

            if not success:
                continue

            delta = new_global_max - current_val

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.15:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.03))
                    if random.random() < prob:
                        accept = True

            if accept:
                current_val = _apply_move_fast(
                    current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                    row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                    pair_idx, max_diff)

                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1

                moved = True
                break

        if not moved:
            no_improve += 1

        T = max(T_min, T * cool)

    return best_rows, best_val

def _exact_descent_polish(current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs,
                          row_maxes, current_val, best_rows, best_val, deadline,
                          pair_idx, max_diff):
    """
    Deterministic local solve: repeatedly scan every (row, position) and apply the
    first improving (or neutral) move found. Continue until a full pass makes no
    change or the deadline is reached. This is a tighter, exact local optimum
    search that complements the stochastic polish.
    """
    progress = True
    passes = 0
    while progress and time.time() < deadline:
        progress = False
        passes += 1
        for ri in range(n_rows):
            if time.time() >= deadline:
                break
            for pi in range(1, k + 1):
                if time.time() >= deadline:
                    break
                row = current_rows[ri]
                old_val = row[pi]
                old_diffs = row_diffs[ri]

                min_allowed = row[pi - 1] + 1
                max_allowed = (row[pi + 1] - 1) if pi < k else max_diff

                # Try decreasing values first (most likely to improve the max)
                for new_val in range(min_allowed, old_val):
                    success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                        current_rows, ri, pi, new_val, static_diffs, row_diffs,
                        all_used_ba, row_maxes, k, max_diff)
                    if not success:
                        continue
                    if new_global_max < current_val:
                        current_val = _apply_move_fast(
                            current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                            row_diffs, all_used_ba, row_maxes, static_diffs, k,
                            old_diffs, pair_idx, max_diff)
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                        progress = True
                        break
                # Also try increasing values (may free up space for other rows)
                # Use a wider range for increasing moves
                inc_limit = min(old_val + 20, max_allowed + 1)
                for new_val in range(old_val + 1, inc_limit):
                    success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                        current_rows, ri, pi, new_val, static_diffs, row_diffs,
                        all_used_ba, row_maxes, k, max_diff)
                    if not success:
                        continue
                    if new_global_max < current_val:
                        current_val = _apply_move_fast(
                            current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                            row_diffs, all_used_ba, row_maxes, static_diffs, k,
                            old_diffs, pair_idx, max_diff)
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                        progress = True
                        break

        # Every 5 passes, try a second round with neutral moves accepted
        # to escape local optima
        if passes % 5 == 0 and time.time() < deadline:
            neutral_progress = False
            for ri in range(n_rows):
                if time.time() >= deadline:
                    break
                for pi in range(1, k + 1):
                    if time.time() >= deadline:
                        break
                    row = current_rows[ri]
                    old_val = row[pi]
                    old_diffs = row_diffs[ri]
                    min_allowed = row[pi - 1] + 1
                    max_allowed = (row[pi + 1] - 1) if pi < k else max_diff
                    for new_val in range(min_allowed, old_val):
                        success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                            current_rows, ri, pi, new_val, static_diffs, row_diffs,
                            all_used_ba, row_maxes, k, max_diff)
                        if not success:
                            continue
                        if new_global_max <= current_val:
                            current_val = _apply_move_fast(
                                current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                                row_diffs, all_used_ba, row_maxes, static_diffs, k,
                                old_diffs, pair_idx, max_diff)
                            if current_val < best_val:
                                best_val = current_val
                                best_rows = [r[:] for r in current_rows]
                            neutral_progress = True
                            break
                    if neutral_progress:
                        progress = True

    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Initialization
    rows = _greedy_init(n_rows, k)

    # Track used differences
    all_used_set = _compute_all_used_set(rows)

    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)

    row_maxes = [max(r) for r in rows]

    # Precompute pair indices and a reasonable max_diff bound for the bytearray.
    # The greedy init gives an upper bound; we allow some headroom for increases.
    greedy_max = best_val
    max_diff = greedy_max + 200  # headroom for random jumps / increases

    pair_idx = _pair_indices(k)

    # 2. Simulated Annealing - Phase 1 (exploration)
    current_rows = [r[:] for r in rows]
    current_val = best_val

    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
    all_used_ba = _compute_all_used_ba(current_rows, max_diff)

    phase1_deadline = start_time + 60.0
    current_rows, current_val, all_used_ba, row_diffs, static_diffs, row_maxes, best_rows, best_val = _sa_phase(
        current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
        current_val, best_rows, best_val, 250.0, 0.5, 0.99994, 30000, phase1_deadline,
        pair_idx, max_diff)

    # 3. Simulated Annealing - Phase 2 (exploitation with lower temp)
    if time.time() < deadline - 20:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used_ba = _compute_all_used_ba(current_rows, max_diff)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]

        phase2_deadline = start_time + 95.0
        current_rows, current_val, all_used_ba, row_diffs, static_diffs, row_maxes, best_rows, best_val = _sa_phase(
            current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
            current_val, best_rows, best_val, 50.0, 0.3, 0.99997, 20000, phase2_deadline,
            pair_idx, max_diff)

    # 4. Stochastic polish phase
    if time.time() < deadline - 10:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used_ba = _compute_all_used_ba(current_rows, max_diff)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]

        polish_deadline = deadline - 5.0
        best_rows, best_val = _polish_phase(
            current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
            current_val, best_rows, best_val, polish_deadline, pair_idx, max_diff)

    # 5. Exact descent polish (deterministic local solve)
    if time.time() < deadline - 2.0:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used_ba = _compute_all_used_ba(current_rows, max_diff)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]

        exact_deadline = deadline - 1.0
        best_rows, best_val = _exact_descent_polish(
            current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
            current_val, best_rows, best_val, exact_deadline, pair_idx, max_diff)

    # 6. Final tight polish with very small tolerance
    if time.time() < deadline - 0.5:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used_ba = _compute_all_used_ba(current_rows, max_diff)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]

        # Very tight final polish
        T = 0.1
        T_min = 0.01
        cool = 0.99999
        no_improve = 0
        max_no_improve_final = 5000

        while time.time() < deadline - 0.2 and no_improve < max_no_improve_final:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)

            old_val = current_rows[ri][pi]
            old_diffs = row_diffs[ri]

            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(15, old_val - min_allowed)

            candidates = []
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)

            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue

            random.shuffle(candidates)

            moved = False
            for new_val in candidates:
                success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                    current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                    row_maxes, k, max_diff)

                if not success:
                    continue

                delta = new_global_max - current_val

                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.05:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.01))
                        if random.random() < prob:
                            accept = True

                if accept:
                    current_val = _apply_move_fast(
                        current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                        row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                        pair_idx, max_diff)

                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1

                    moved = True
                    break

            if not moved:
                no_improve += 1

            T = max(T_min, T * cool)

    # Final validation and return
    return best_rows

# EVOLVE-BLOCK-END
