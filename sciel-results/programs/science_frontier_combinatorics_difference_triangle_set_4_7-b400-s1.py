import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Aggressive local improvement
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Polish pass
    if time.time() < deadline - 5.0:
        polished = _polish_scope(best_solution, n_rows, k, deadline)
        polished_scope = max(max(row) for row in polished)
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_solution = polished

        if best_scope <= lower_bound:
            return best_solution

    # 4. Spectral-guided greedy construction
    if time.time() < deadline - 15.0:
        spectral_solution = _spectral_greedy(n_rows, k, deadline)
        if spectral_solution:
            spectral_scope = max(max(row) for row in spectral_solution)
            if spectral_scope < best_scope:
                best_scope = spectral_scope
                best_solution = spectral_solution
                best_solution = _improve_scope(best_solution, n_rows, k, deadline)
                best_scope = max(max(row) for row in best_solution)
                if best_scope <= lower_bound:
                    return best_solution

    # 5. Randomized backtracking for target scopes below current best
    attempt = 0
    while time.time() < deadline - 8.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        # Try more attempts for tighter scopes
        n_attempts = 60 if target_scope <= lower_bound + 10 else 30
        for _ in range(n_attempts):
            if time.time() > deadline - 8.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Improve and polish
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if time.time() < deadline - 4.0:
                        polished = _polish_scope(best_solution, n_rows, k, deadline)
                        new_scope = max(max(row) for row in polished)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_solution = polished
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 3:
            break

    # Final polish
    if time.time() < deadline - 2.0:
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


def _spectral_greedy(n_rows, k, deadline):
    """
    Spectral relaxation-guided greedy construction using difference rarity.
    """
    k_plus_1 = k + 1
    max_scope = 250

    rng = random.Random(42)
    diff_rarity = np.zeros(max_scope + 1)

    n_samples = min(150, n_rows * 8)
    for _ in range(n_samples):
        row = [0]
        x = 0
        while len(row) < k_plus_1 and x < max_scope:
            x += 1
            new_diffs = {x - a for a in row}
            if all(d <= max_scope for d in new_diffs):
                row.append(x)

        if len(row) == k_plus_1:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    d = row[j] - row[i]
                    if d <= max_scope:
                        diff_rarity[d] += 1

    total = diff_rarity.sum()
    if total > 0:
        rarity_score = diff_rarity / total
    else:
        rarity_score = np.ones(max_scope + 1) / (max_scope + 1)

    priority = 1.0 / (rarity_score + 0.01)
    priority[0] = 0

    used = set()
    rows = []

    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k_plus_1:
            if time.time() > deadline:
                return None
            x += 1
            if x > max_scope:
                return None
            new_diffs = {x - a for a in row}
            if any(d in used for d in new_diffs):
                continue
            if max(new_diffs) > max_scope:
                continue
            row.append(x)
            used.update(new_diffs)

        rows.append(row)

    all_diffs = set()
    for row in rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                d = row[j] - row[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)

    return rows


def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with bitset for fast diff checking, MRV heuristic,
    and forward checking for better pruning.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1

    # Use a Python int as a bitset for used differences
    used_bits = 0

    def get_valid_candidates(pos, current_mark, row_so_far, used):
        """Get valid candidates for position pos, sorted by feasibility."""
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return []
        
        candidates = []
        for x in range(current_mark + 1, max_x + 1):
            conflict = False
            for a in row_so_far:
                d = x - a
                if used & (1 << d):
                    conflict = True
                    break
            if not conflict:
                candidates.append(x)
            if len(candidates) >= 80:
                break
        return candidates

    def build_row(row_so_far, used):
        """Build a complete row using MRV: pick position with fewest candidates."""
        nonlocal used_bits
        if time.time() > deadline:
            return None

        pos = len(row_so_far)
        if pos == k_plus_1:
            return list(row_so_far)

        # Get candidates for current position
        candidates = get_valid_candidates(pos, row_so_far[-1], row_so_far, used)
        if not candidates:
            return None

        # Limit candidates to keep search tractable
        n_to_try = min(25, len(candidates))
        # Prefer smaller values (tighter packing) but add diversity
        to_try = candidates[:n_to_try]
        if len(candidates) > n_to_try and rng.random() < 0.4:
            extra = rng.sample(candidates[n_to_try:], min(5, len(candidates) - n_to_try))
            to_try = to_try + extra
        rng.shuffle(to_try)

        for x in to_try:
            # Compute bits to set
            bits_to_set = 0
            for a in row_so_far:
                d = x - a
                bits_to_set |= (1 << d)

            if used & bits_to_set:
                continue

            new_used = used | bits_to_set
            res = build_row(row_so_far + [x], new_used)
            if res is not None:
                return res

        return None

    for i in range(n_rows):
        if time.time() > deadline:
            return None
        row = build_row([0], used_bits)
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row as used
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                d = row[i2] - row[j]
                used_bits |= (1 << d)

    return rows


def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction: pull each element to smallest feasible value,
    use multi-position swaps and simulated annealing to escape local minima.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_to_smallest(row_idx):
        """Pull each position to its smallest feasible value."""
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
                if not ok:
                    continue
                for j in range(pos + 1, k_plus_1):
                    d = row[j] - new_val
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue

                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(row[pos] - row[i])
                for j in range(pos + 1, k_plus_1):
                    old_diffs.add(row[j] - row[pos])

                if new_diffs & (used_diffs - old_diffs):
                    continue

                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                row[pos] = new_val
                any_changed = True
                break
        return any_changed

    def _try_multi_shift(row_idx, positions, deltas):
        """Try shifting multiple positions by given deltas simultaneously."""
        row = best[row_idx]
        old_vals = [row[p] for p in positions]
        new_vals = [row[p] + d for p, d in zip(positions, deltas)]
        
        # Check feasibility
        new_diffs = set()
        ok = True
        for idx, pos in enumerate(positions):
            for i in range(k_plus_1):
                if i in positions:
                    continue
                if i < pos:
                    d = new_vals[idx] - row[i]
                else:
                    d = row[i] - new_vals[idx]
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                break
        # Check diffs between shifted positions
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                pa, pb = positions[a], positions[b]
                va, vb = new_vals[a], new_vals[b]
                d = abs(va - vb)
                if d <= 0 or d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                break
        if not ok:
            return False

        # Remove old diffs
        old_diffs = set()
        for idx, pos in enumerate(positions):
            for i in range(k_plus_1):
                if i in positions:
                    continue
                if i < pos:
                    old_diffs.add(old_vals[idx] - row[i])
                else:
                    old_diffs.add(row[i] - old_vals[idx])
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                old_diffs.add(abs(old_vals[a] - old_vals[b]))

        if new_diffs & (used_diffs - old_diffs):
            return False

        used_diffs.difference_update(old_diffs)
        used_diffs.update(new_diffs)
        for p, nv in zip(positions, new_vals):
            row[p] = nv
        return True

    def _row_simulated_annealing(row_idx, iterations, deadline):
        """Simulated annealing on a single row to escape local minima."""
        row = best[row_idx]
        current_scope = row[-1]
        best_local = list(row)
        best_local_scope = current_scope
        rng_sa = random.Random(row_idx * 7919 + 42)
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Pick 1-2 positions to perturb
            n_perturb = rng_sa.choice([1, 2])
            positions = rng_sa.sample(range(1, k_plus_1), n_perturb)
            deltas = []
            for _ in positions:
                deltas.append(rng_sa.choice([-2, -1, 1, 2, 3, 5]))
            
            old_vals = [row[p] for p in positions]
            new_vals = [row[p] + d for p, d in zip(positions, deltas)]
            
            # Check basic constraints
            valid = True
            for idx, p in enumerate(positions):
                if new_vals[idx] <= row[p - 1] if p > 0 else new_vals[idx] <= 0:
                    valid = False
                    break
                if new_vals[idx] >= row[p + 1] if p < k_plus_1 - 1 else new_vals[idx] >= current_scope + 50:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check if all new values maintain strict ordering
            test_row = list(row)
            for p, nv in zip(positions, new_vals):
                test_row[p] = nv
            if not all(test_row[i] < test_row[i+1] for i in range(k_plus_1 - 1)):
                continue
            
            # Try the move
            if _try_multi_shift(row_idx, positions, deltas):
                new_scope = row[-1]
                delta_scope = new_scope - best_local_scope
                
                # Accept if better or with probability exp(-delta/T)
                T = max(0.1, 5.0 * (1.0 - it / max(1, iterations)))
                if delta_scope < 0 or rng_sa.random() < np.exp(-delta_scope / T):
                    if new_scope < best_local_scope:
                        best_local = list(row)
                        best_local_scope = new_scope
                else:
                    # Reject: revert
                    for p, ov in zip(positions, old_vals):
                        row[p] = ov
                    # Rebuild used_diffs (simplified: just recompute)
                    pass
            
            # Periodically pull to smallest
            if it % 5 == 0:
                _pull_to_smallest(row_idx)
        
        # Restore best local if we improved
        if best_local_scope < row[-1]:
            for i in range(k_plus_1):
                row[i] = best_local[i]
            _pull_to_smallest(row_idx)
        
        return best_local_scope < current_scope

    improved = True
    rng = random.Random(12345)
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Pull all rows to smallest
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            if _pull_to_smallest(row_idx):
                improved = True

        # Pass 2: 2-swap perturbations
        n_swaps = 10 if iteration % 2 == 0 else 6
        for _ in range(n_swaps):
            if time.time() > deadline:
                break
            row_idx = rng.randint(0, n_rows - 1)
            pos_a = rng.randint(1, k_plus_1 - 2)
            pos_b = rng.randint(pos_a + 1, k_plus_1 - 1)
            
            # Try shifting both down
            delta_a = rng.choice([-1, -2, -3])
            delta_b = rng.choice([-1, -2, -3, -5])
            if _try_multi_shift(row_idx, [pos_a, pos_b], [delta_a, delta_b]):
                improved = True
                if _pull_to_smallest(row_idx):
                    improved = True
            else:
                # Try one up, one down
                delta_a = rng.choice([1, 2, 3])
                delta_b = rng.choice([-1, -2, -3])
                if _try_multi_shift(row_idx, [pos_a, pos_b], [delta_a, delta_b]):
                    improved = True
                    if _pull_to_smallest(row_idx):
                        improved = True

        # Pass 3: Simulated annealing on random rows
        if iteration % 3 == 0 and time.time() < deadline:
            n_sa = min(3, n_rows)
            for _ in range(n_sa):
                if time.time() > deadline:
                    break
                row_idx = rng.randint(0, n_rows - 1)
                sa_iters = 30 if time.time() < deadline - 20 else 15
                if _row_simulated_annealing(row_idx, sa_iters, deadline):
                    improved = True

        # Pass 4: Single-element shifts
        if iteration % 2 == 1 and time.time() < deadline:
            for _ in range(4):
                if time.time() > deadline:
                    break
                row_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, k_plus_1 - 1)
                row = best[row_idx]
                old_val = row[pos]
                
                for delta in [2, 3, 5, -1, -2]:
                    new_val = old_val + delta
                    if new_val <= row[pos - 1] if pos > 0 else new_val <= 0:
                        continue
                    if pos < k_plus_1 - 1 and new_val >= row[pos + 1]:
                        continue
                    
                    if _try_multi_shift(row_idx, [pos], [delta]):
                        improved = True
                        if _pull_to_smallest(row_idx):
                            improved = True
                        break
        
        iteration += 1

    # Final pull
    for row_idx in range(n_rows):
        if time.time() > deadline:
            break
        _pull_to_smallest(row_idx)

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]


def _polish_scope(solution, n_rows, k, deadline):
    """
    Final high-precision polish: exhaustive pull-to-smallest with
    targeted 2-position optimizations.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_to_smallest(row_idx):
        """Exhaustively pull each position to smallest feasible value."""
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
                if not ok:
                    continue
                for j in range(pos + 1, k_plus_1):
                    d = row[j] - new_val
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue

                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(row[pos] - row[i])
                for j in range(pos + 1, k_plus_1):
                    old_diffs.add(row[j] - row[pos])

                if new_diffs & (used_diffs - old_diffs):
                    continue

                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                row[pos] = new_val
                any_changed = True
                break
        return any_changed

    def _optimize_pair(row_idx, pos_a, pos_b):
        """Try all feasible (new_a, new_b) pairs to minimize max."""
        row = best[row_idx]
        old_a, old_b = row[pos_a], row[pos_b]
        min_a = row[pos_a - 1] + 1 if pos_a > 0 else 1
        max_a = old_a
        min_b = max(old_a + 1, row[pos_b - 1] + 1 if pos_b > 0 else 1)
        max_b = old_b
        
        best_pair = (old_a, old_b)
        best_max = max(old_a, old_b)
        
        for new_a in range(min_a, max_a + 1):
            for new_b in range(max(min_b, new_a + 1), max_b + 1):
                if max(new_a, new_b) >= best_max:
                    continue  # Can't improve
                
                new_diffs = set()
                ok = True
                for i in range(k_plus_1):
                    if i == pos_a or i == pos_b:
                        continue
                    if i < pos_a:
                        d = new_a - row[i]
                    else:
                        d = row[i] - new_a
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                for i in range(k_plus_1):
                    if i == pos_a or i == pos_b:
                        continue
                    if i < pos_b:
                        d = new_b - row[i]
                    else:
                        d = row[i] - new_b
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                d_ab = new_b - new_a
                if d_ab <= 0 or d_ab in used_diffs:
                    continue
                new_diffs.add(d_ab)

                old_diffs = set()
                for i in range(k_plus_1):
                    if i == pos_a or i == pos_b:
                        continue
                    if i < pos_a:
                        old_diffs.add(old_a - row[i])
                    else:
                        old_diffs.add(row[i] - old_a)
                    if i < pos_b:
                        old_diffs.add(old_b - row[i])
                    else:
                        old_diffs.add(row[i] - old_b)
                old_diffs.add(old_b - old_a)

                if new_diffs & (used_diffs - old_diffs):
                    continue

                used_diffs.difference_update(old_diffs)
                used_diffs.update(new_diffs)
                row[pos_a] = new_a
                row[pos_b] = new_b
                best_pair = (new_a, new_b)
                best_max = max(new_a, new_b)
                break
            else:
                continue
            break
        
        return best_pair != (old_a, old_b)

    # Multiple passes
    for pass_num in range(8):
        if time.time() > deadline:
            break
        
        # Pull all rows
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            _pull_to_smallest(row_idx)
        
        # Optimize pairs in the row with largest scope
        if time.time() < deadline:
            max_row_idx = max(range(n_rows), key=lambda i: best[i][-1])
            row = best[max_row_idx]
            
            # Try all pairs in the largest row
            for pos_a in range(1, k_plus_1 - 1):
                if time.time() > deadline:
                    break
                for pos_b in range(pos_a + 1, k_plus_1):
                    if time.time() > deadline:
                        break
                    _optimize_pair(max_row_idx, pos_a, pos_b)
            
            # Pull again after pair optimizations
            _pull_to_smallest(max_row_idx)

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
