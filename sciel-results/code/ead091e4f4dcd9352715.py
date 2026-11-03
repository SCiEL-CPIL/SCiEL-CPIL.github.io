import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0

    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..M, so max entry >= num_diffs - 1

    k_plus_1 = k + 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Polish pass for further compaction
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Try to find better solutions with randomized backtracking
    # for target scopes below current best
    attempt = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        # Try multiple random restarts for this target scope
        found = False
        for _ in range(80):
            if time.time() > deadline - 5.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Try to improve further
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    # Polish
                    polished = _polish_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in polished)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = polished
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 5:
            # If we can't find anything near the lower bound, stop
            if target_scope < lower_bound:
                break

    # Final polish
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    return best_solution


def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
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


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with MRV heuristic, bitset tracking, and better pruning.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    max_diff = max_scope
    used_bits = 0  # Python int as arbitrary-precision bitset
    node_count = [0]

    def build_row(pos, current_mark, row_so_far, depth):
        nonlocal used_bits
        if time.time() > deadline:
            return None

        node_count[0] += 1
        if node_count[0] % 100000 == 0 and time.time() > deadline - 0.1:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Prune upper bound: need at least (k_plus_1 - pos - 1) more elements
        remaining = k_plus_1 - pos - 1
        max_x = max_scope - remaining
        if max_x <= current_mark:
            return None

        # Find valid candidates for current position
        valid_candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if (used_bits >> d) & 1:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
            if len(valid_candidates) >= 40:  # Limit for speed
                break

        if not valid_candidates:
            return None

        # Feasibility check: need enough candidates for remaining positions
        if len(valid_candidates) < remaining + 1:
            return None

        # Strategy: Try smallest candidates first (greedy), with some randomness
        n_smallest = min(15, len(valid_candidates))
        candidates_to_try = valid_candidates[:n_smallest]

        # Occasionally try larger candidates to escape local minima
        if len(valid_candidates) > n_smallest and rng.random() < 0.4:
            extra_count = min(8, len(valid_candidates) - n_smallest)
            extra = rng.sample(valid_candidates[n_smallest:], extra_count)
            candidates_to_try = candidates_to_try + extra

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Mark new differences using bit operations
            mask = 0
            for a in row_so_far:
                d = x - a
                mask |= (1 << d)
            old_bits = used_bits
            used_bits |= mask

            res = build_row(pos + 1, x, row_so_far + [x], depth + 1)
            if res is not None:
                return res

            # Unmark differences
            used_bits = old_bits

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0], 0)
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row as used
        mask = 0
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                mask |= (1 << d)
        used_bits |= mask

    return rows


def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction with simulated annealing for better local minima escape.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_to_smallest(row_idx):
        """Pull all elements in row row_idx to their smallest feasible values."""
        row = best[row_idx]
        any_changed = False
        for pos in range(1, k_plus_1):
            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                new_diffs = set()
                ok = True
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                if ok:
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(row[pos] - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - row[pos])
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_changed = True
                    break
        return any_changed

    def _try_move(row_idx, pos, new_val):
        """Try moving element at (row_idx, pos) to new_val. Returns True if successful."""
        row = best[row_idx]
        old_val = row[pos]
        if new_val == old_val:
            return False
        if new_val <= row[pos - 1] or (pos < k_plus_1 - 1 and new_val >= row[pos + 1]):
            return False

        new_diffs = set()
        for i in range(pos):
            d = new_val - row[i]
            if d <= 0 or d in used_diffs:
                return False
            new_diffs.add(d)
        for j in range(pos + 1, k_plus_1):
            d = row[j] - new_val
            if d <= 0 or d in used_diffs:
                return False
            new_diffs.add(d)

        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, k_plus_1):
            old_diffs.add(row[j] - old_val)

        if new_diffs & (used_diffs - old_diffs):
            return False

        used_diffs.difference_update(old_diffs)
        used_diffs.update(new_diffs)
        row[pos] = new_val
        return True

    # Phase 1: Pull to smallest
    for iteration in range(30):
        if time.time() > deadline - 5.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Phase 2: Simulated annealing with more steps and better cooling
    import math
    current_solution = [list(row) for row in best]
    current_scope = max(max(row) for row in current_solution)
    best_sa = [list(row) for row in current_solution]
    best_sa_scope = current_scope

    # Rebuild used_diffs for current_solution
    used_diffs_sa = set()
    for row in current_solution:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs_sa.add(row[j] - row[i])

    def _try_move_sa(row_idx, pos, new_val):
        """Try move in SA context."""
        row = current_solution[row_idx]
        old_val = row[pos]
        if new_val == old_val:
            return False
        if new_val <= row[pos - 1] or (pos < k_plus_1 - 1 and new_val >= row[pos + 1]):
            return False

        new_diffs = set()
        for i in range(pos):
            d = new_val - row[i]
            if d <= 0 or d in used_diffs_sa:
                return False
            new_diffs.add(d)
        for j in range(pos + 1, k_plus_1):
            d = row[j] - new_val
            if d <= 0 or d in used_diffs_sa:
                return False
            new_diffs.add(d)

        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, k_plus_1):
            old_diffs.add(row[j] - old_val)

        if new_diffs & (used_diffs_sa - old_diffs):
            return False

        used_diffs_sa.difference_update(old_diffs)
        used_diffs_sa.update(new_diffs)
        row[pos] = new_val
        return True

    def _undo_move_sa(row_idx, pos, old_val):
        """Undo a move in SA context."""
        row = current_solution[row_idx]
        new_val = row[pos]
        row[pos] = old_val

        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, k_plus_1):
            new_diffs.add(row[j] - new_val)

        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, k_plus_1):
            old_diffs.add(row[j] - old_val)

        used_diffs_sa.difference_update(new_diffs)
        used_diffs_sa.update(old_diffs)

    def _try_multi_move_sa(row_idx, moves):
        """Try multiple moves simultaneously. moves is list of (pos, new_val)."""
        row = current_solution[row_idx]
        # Check all moves are valid simultaneously
        all_new_diffs = set()
        all_old_diffs = set()
        for pos, new_val in moves:
            old_val = row[pos]
            if new_val == old_val:
                continue
            if new_val <= row[pos - 1] or (pos < k_plus_1 - 1 and new_val >= row[pos + 1]):
                return False
            for i in range(pos):
                d = new_val - row[i]
                if d <= 0 or d in used_diffs_sa:
                    return False
                all_new_diffs.add(d)
            for j in range(pos + 1, k_plus_1):
                d = row[j] - new_val
                if d <= 0 or d in used_diffs_sa:
                    return False
                all_new_diffs.add(d)
            for i in range(pos):
                all_old_diffs.add(old_val - row[i])
            for j in range(pos + 1, k_plus_1):
                all_old_diffs.add(row[j] - old_val)
        
        if all_new_diffs & (used_diffs_sa - all_old_diffs):
            return False

        used_diffs_sa.difference_update(all_old_diffs)
        used_diffs_sa.update(all_new_diffs)
        for pos, new_val in moves:
            row[pos] = new_val
        return True

    def _undo_multi_move_sa(row_idx, old_states):
        """Undo multiple moves. old_states is list of (pos, old_val)."""
        row = current_solution[row_idx]
        for pos, old_val in old_states:
            new_val = row[pos]
            row[pos] = old_val
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, k_plus_1):
                new_diffs.add(row[j] - new_val)
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, k_plus_1):
                old_diffs.add(row[j] - old_val)
            used_diffs_sa.difference_update(new_diffs)
            used_diffs_sa.update(old_diffs)

    # Adaptive SA: run multiple passes with different temperatures
    for sa_pass in range(4):
        total_steps = 500
        start_temp = 10.0 if sa_pass == 0 else (6.0 if sa_pass == 1 else (3.0 if sa_pass == 2 else 1.0))
        end_temp = 0.03
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, total_steps - 1))

        for step in range(total_steps):
            if time.time() > deadline - 5.0:
                break

            temp = start_temp * (cooling_rate ** step)

            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k_plus_1 - 1)
            row = current_solution[row_idx]
            old_val = row[pos]

            # Bias toward pulling values down (reduces scope)
            if random.random() < 0.7:
                # Try to pull down
                min_val = row[pos - 1] + 1
                max_val = row[pos] - 1
                if max_val >= min_val:
                    new_val = random.randint(min_val, max_val)
                else:
                    new_val = old_val - random.randint(1, 3)
            else:
                # Occasionally push up to escape local minima
                offset = random.choice([1, 2, 3, 5, 7, 10])
                new_val = old_val + offset

            if _try_move_sa(row_idx, pos, new_val):
                new_scope = max(max(r) for r in current_solution)
                delta = new_scope - current_scope

                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    current_scope = new_scope
                    if new_scope < best_sa_scope:
                        best_sa_scope = new_scope
                        best_sa = [list(r) for r in current_solution]
                else:
                    _undo_move_sa(row_idx, pos, old_val)

            # Occasionally try multi-position moves to escape local minima
            if step % 20 == 0 and random.random() < 0.3:
                n_moves = random.randint(2, min(4, k_plus_1 - 1))
                positions = random.sample(range(1, k_plus_1), n_moves)
                moves = []
                old_states = []
                for p in sorted(positions):
                    old_v = current_solution[row_idx][p]
                    min_v = current_solution[row_idx][p - 1] + 1
                    max_v = old_v - 1
                    if max_v >= min_v:
                        new_v = random.randint(min_v, max_v)
                    else:
                        new_v = old_v + random.randint(1, 5)
                    moves.append((p, new_v))
                    old_states.append((p, old_v))
                
                if _try_multi_move_sa(row_idx, moves):
                    new_scope = max(max(r) for r in current_solution)
                    delta = new_scope - current_scope
                    if delta <= 0 or random.random() < math.exp(-delta / temp):
                        current_scope = new_scope
                        if new_scope < best_sa_scope:
                            best_sa_scope = new_scope
                            best_sa = [list(r) for r in current_solution]
                    else:
                        _undo_multi_move_sa(row_idx, old_states)

    # Update best if SA found improvement
    if best_sa_scope < best_scope:
        best = best_sa
        best_scope = best_sa_scope

    # Phase 3: Final pull pass with iterative refinement
    # Rebuild used_diffs for best
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    # Multiple rounds of pull with occasional small perturbations to escape local minima
    for iteration in range(40):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            # Try a small perturbation to escape local minimum
            if iteration % 10 == 9 and iteration < 30:
                row_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k_plus_1 - 1)
                old_val = best[row_idx][pos]
                new_val = old_val + random.randint(1, 3)
                if _try_move(row_idx, pos, new_val):
                    # Rebuild used_diffs after perturbation
                    used_diffs = set()
                    for row in best:
                        for i in range(k_plus_1):
                            for j in range(i + 1, k_plus_1):
                                used_diffs.add(row[j] - row[i])
                    # Pull again
                    for r in range(n_rows):
                        _pull_to_smallest(r)
                    # Rebuild used_diffs again
                    used_diffs = set()
                    for row in best:
                        for i in range(k_plus_1):
                            for j in range(i + 1, k_plus_1):
                                used_diffs.add(row[j] - row[i])
                else:
                    break
            else:
                break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]


def _polish_scope(solution, n_rows, k, deadline):
    """
    High-precision polish with multiple SA restarts and final greedy pull.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_all():
        """Pull all rows to smallest feasible values."""
        any_changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(1, k_plus_1):
                for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                    new_diffs = set()
                    ok = True
                    for i in range(pos):
                        d = new_val - row[i]
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        for j in range(pos + 1, k_plus_1):
                            d = row[j] - new_val
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                    if ok:
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(row[pos] - row[i])
                        for j in range(pos + 1, k_plus_1):
                            old_diffs.add(row[j] - row[pos])
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        any_changed = True
                        break
        return any_changed

    def _sa_pass(current, current_scope, max_steps=400):
        """Run a simulated annealing pass."""
        import math
        sol = [list(r) for r in current]
        scope = current_scope

        used = set()
        for row in sol:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    used.add(row[j] - row[i])

        def try_move(row_idx, pos, new_val):
            row = sol[row_idx]
            old_val = row[pos]
            if new_val == old_val:
                return False
            if new_val <= row[pos - 1] or (pos < k_plus_1 - 1 and new_val >= row[pos + 1]):
                return False
            new_diffs = set()
            for i in range(pos):
                d = new_val - row[i]
                if d <= 0 or d in used:
                    return False
                new_diffs.add(d)
            for j in range(pos + 1, k_plus_1):
                d = row[j] - new_val
                if d <= 0 or d in used:
                    return False
                new_diffs.add(d)
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, k_plus_1):
                old_diffs.add(row[j] - old_val)
            if new_diffs & (used - old_diffs):
                return False
            used.difference_update(old_diffs)
            used.update(new_diffs)
            row[pos] = new_val
            return True

        def undo_move(row_idx, pos, old_val):
            row = sol[row_idx]
            new_val = row[pos]
            row[pos] = old_val
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, k_plus_1):
                new_diffs.add(row[j] - new_val)
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, k_plus_1):
                old_diffs.add(row[j] - old_val)
            used.difference_update(new_diffs)
            used.update(old_diffs)

        start_temp = 6.0
        end_temp = 0.03
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_steps - 1))

        for step in range(max_steps):
            if time.time() > deadline - 3.0:
                break
            temp = start_temp * (cooling_rate ** step)
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k_plus_1 - 1)
            row = sol[row_idx]
            old_val = row[pos]
            offset = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 10])
            new_val = old_val + offset
            if try_move(row_idx, pos, new_val):
                new_scope = max(max(r) for r in sol)
                delta = new_scope - scope
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    scope = new_scope
                else:
                    undo_move(row_idx, pos, old_val)
        return sol, scope

    # Initial pull
    _pull_all()
    best_scope = max(max(row) for row in best)

    # Multiple SA restarts with more aggressive perturbation
    for restart in range(6):
        if time.time() > deadline - 3.0:
            break
        # Perturb current best
        perturbed = [list(row) for row in best]
        for _ in range(10):
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k_plus_1 - 1)
            offset = random.choice([2, 3, 5, 7, 10, 15])
            perturbed[row_idx][pos] += offset
            # Re-pull to fix
            _pull_all()

        sol, scope = _sa_pass(perturbed, max(max(row) for row in perturbed), max_steps=500)
        if scope < best_scope:
            best = sol
            best_scope = scope
            # Rebuild used_diffs
            used_diffs = set()
            for row in best:
                for i in range(k_plus_1):
                    for j in range(i + 1, k_plus_1):
                        used_diffs.add(row[j] - row[i])
            _pull_all()
            best_scope = max(max(row) for row in best)

    # Final high-precision polish: iterate pull until no improvement
    for final_iter in range(30):
        if time.time() > deadline - 1.0:
            break
        if not _pull_all():
            break
        best_scope = max(max(row) for row in best)

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
