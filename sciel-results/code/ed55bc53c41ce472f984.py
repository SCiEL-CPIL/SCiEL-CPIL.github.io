import time
import random
import sys
import math

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Generate initial solution using fast greedy construction
    best_solution = _greedy_construct_fast(n_rows, k)
    if not best_solution:
        best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)

    # Phase 1: Simulated Annealing with static-diff optimization
    improved = _simulated_annealing(best_solution, n_rows, k, deadline - 35)
    if improved:
        curr_score = max(max(r) for r in improved)
        if curr_score < best_score:
            best_score = curr_score
            best_solution = improved

    # Phase 2: Iterated Local Search with perturbations
    no_improve_count = 0
    while time.time() < deadline - 25:
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]

        # Choose perturbation strategy based on stagnation
        if no_improve_count >= 4:
            # Strong perturbation: rebuild 1-2 rows greedily
            n_rebuild = min(2, n_rows)
            for _ in range(n_rebuild):
                r_idx = random.randint(0, n_rows - 1)
                other_rows = [r for i, r in enumerate(candidate) if i != r_idx]
                used_diffs_other = _compute_all_diffs(other_rows)
                new_row = _greedy_row_given_diffs(used_diffs_other, k)
                if new_row is not None:
                    candidate[r_idx] = new_row
            no_improve_count = 0
        else:
            # Mild perturbation: several random valid moves
            n_moves = random.randint(2, 5)
            for _ in range(n_moves):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
                if new_val is not None and new_val != candidate[r_idx][pos]:
                    candidate[r_idx][pos] = new_val

        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline - 20)

        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

    # Phase 3: Final high-precision polish with remaining time
    if time.time() < deadline - 2:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline - 1)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution


def _greedy_row_given_diffs(used_diffs, k):
    """Build a single row greedily given already-used differences."""
    row = [0]
    current_max = 0
    while len(row) < k + 1:
        found = False
        for x in range(current_max + 1, current_max + 2000):
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break

            if valid:
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                current_max = x
                found = True
                break
        if not found:
            return None
    return row

def _row_diffs(row):
    """Return the set of all positive differences within a single row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s

def _compute_all_diffs(rows):
    """Return the union of all positive differences across all rows."""
    all_diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                all_diffs.add(r[j] - ri)
    return all_diffs

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """Find a random valid value for rows[r_idx][pos] different from current."""
    row = rows[r_idx]
    old_val = row[pos]
    n = len(row)

    # Compute all diffs once
    all_diffs = _compute_all_diffs(rows)

    # Diffs involving the old value at this position
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)

    forbidden_diffs = all_diffs - diffs_involving_old

    # Search for a new value
    lower_bound = row[pos - 1] + 1
    current_max = max(max(r) for r in rows)
    upper_bound = min(current_max + 30, 250)
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos + 1] - 1)

    if upper_bound < lower_bound:
        return None

    # Collect valid candidates, biasing toward lower values
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue

        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val

            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)

        if valid:
            valid_candidates.append(new_val)

    if not valid_candidates:
        return None

    # Weighted random choice favoring smaller values
    if len(valid_candidates) == 1:
        return valid_candidates[0]

    max_candidate = max(valid_candidates)
    total_weight = 0
    weights = []
    for v in valid_candidates:
        weight = (max_candidate - v + 1) ** 2
        weights.append(weight)
        total_weight += weight

    r = random.uniform(0, total_weight)
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return valid_candidates[i]
    return valid_candidates[-1]

def _build_static_diffs(row, k):
    """
    For each position pi (1..k), compute the set of differences in the row
    that do NOT involve position pi.
    """
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                s.add(row[j] - row[i])
        static[pi] = s
    return static

def _compute_new_diffs_fast(row, pi, new_val, static_diffs_pi):
    """
    Given a row with position pi replaced by new_val, compute the full diff set
    using the precomputed static diffs plus the k diffs involving pi.
    Returns (new_diffs_set, valid_internal).
    """
    n = len(row)
    new_diffs = set(static_diffs_pi)
    valid = True
    for i in range(n):
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
    return new_diffs, valid

def _simulated_annealing(rows, n_rows, k, deadline):
    """Simulated annealing with static-diff optimization for fast conflict checking."""
    current_rows = [r[:] for r in rows]
    all_used = _compute_all_diffs(current_rows)
    
    row_diffs = [_row_diffs(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]
    
    current_val = max(row_maxes)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    T = 150.0
    T_min = 0.3
    cool = 0.99997
    
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline - 2 and no_improve < max_no_improve:
        # Pick a row and position to modify
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        
        # Generate candidates with bias toward reducing max
        candidates = []
        
        # Decreasing candidates - try more of these
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(40, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            new_v = old_val - d
            candidates.append(new_v)
            
        # Increasing candidates (for exploration) - fewer
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 25
        max_increase = min(12, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            new_v = old_val + d
            candidates.append(new_v)
        
        # Random jump (large perturbation)
        if random.random() < 0.12:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
            
        random.shuffle(candidates)
        
        # Pre-fetch static diffs for this position
        static_pi = static_diffs[ri][pi]
        
        moved = False
        for new_val in candidates:
            # Basic validity check for this position
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
                
            # Fast diff computation using static diffs
            new_diffs, valid_internal = _compute_new_diffs_fast(
                current_rows[ri], pi, new_val, static_pi)
            
            if not valid_internal:
                continue
                
            # Get old diffs of row ri
            old_diffs = row_diffs[ri]
            
            # Check for conflicts: new_diffs should not intersect with (all_used - old_diffs)
            conflict = False
            for d in new_diffs:
                if d in all_used and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Move is valid - compute new global max efficiently
            if pi == k:
                cand_row_max = new_val
            else:
                cand_row_max = current_rows[ri][k]
            
            # Update row_maxes temporarily
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max  # restore
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.15:
                    accept = True
            else:
                # Accept worse moves with probability
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                all_used -= old_diffs
                all_used |= new_diffs
                current_rows[ri][pi] = new_val
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = new_global_max
                
                # Rebuild static diffs for this row
                static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                
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
        
        # Periodic restart if stuck for a very long time
        if no_improve > 25000 and random.random() < 0.05:
            # Perturb and restart
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                new_val = _find_random_valid_value(current_rows, r_idx, pos, n_rows, k)
                if new_val is not None and new_val != current_rows[r_idx][pos]:
                    current_rows[r_idx][pos] = new_val
                    # Recompute all state
                    all_used = _compute_all_diffs(current_rows)
                    row_diffs = [_row_diffs(r) for r in current_rows]
                    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
                    row_maxes = [max(r) for r in current_rows]
                    current_val = max(row_maxes)
            no_improve = 0
            T = max(T, 50.0)

    return best_rows

def _local_search_minimize(rows, n_rows, k, deadline):
    """Multi-pass local search to minimize max entry with optimized diff tracking."""
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)

    while time.time() < deadline - 2:
        progress = False

        # Pass 1: Right-to-left greedy lowering per row
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 2:
                break

            row = best_rows[r_idx]
            n = len(row)

            for pos in range(k, 0, -1):
                old_val = row[pos]

                # Compute diffs involving old value
                diffs_involving_old = set()
                for i in range(n):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)

                # Forbidden diffs are all used diffs except those involving old value
                forbidden_diffs = used_diffs - diffs_involving_old

                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < n:
                    upper_bound = min(upper_bound, row[pos + 1] - 1)

                if upper_bound < lower_bound:
                    continue

                # Try to find a smaller valid value
                found = None
                new_diffs = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    nd = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val

                        if d in forbidden_diffs or d in nd:
                            valid = False
                            break
                        nd.add(d)

                    if valid:
                        found = new_val
                        new_diffs = nd
                        break

                if found is not None and found < old_val:
                    # Update used_diffs: remove old diffs, add new diffs
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True

        # Pass 2: Left-to-right to catch cascading improvements
        if time.time() < deadline - 2:
            for r_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break

                row = best_rows[r_idx]
                n = len(row)

                for pos in range(1, k + 1):
                    old_val = row[pos]

                    diffs_involving_old = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            diffs_involving_old.add(old_val - row[i])
                        else:
                            diffs_involving_old.add(row[i] - old_val)

                    forbidden_diffs = used_diffs - diffs_involving_old

                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1
                    if pos + 1 < n:
                        upper_bound = min(upper_bound, row[pos + 1] - 1)

                    if upper_bound < lower_bound:
                        continue

                    found = None
                    new_diffs = None
                    for new_val in range(lower_bound, upper_bound + 1):
                        valid = True
                        nd = set()
                        for i in range(n):
                            if i == pos:
                                continue
                            if i < pos:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val

                            if d in forbidden_diffs or d in nd:
                                valid = False
                                break
                            nd.add(d)

                        if valid:
                            found = new_val
                            new_diffs = nd
                            break

                    if found is not None and found < old_val:
                        for d in diffs_involving_old:
                            used_diffs.discard(d)
                        for d in new_diffs:
                            used_diffs.add(d)
                        row[pos] = found
                        progress = True

        if not progress:
            break

    return best_rows

def _greedy_construct(n_rows, k):
    """Greedy baseline construction."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 2000):
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break

                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used_diffs.add(x - a)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows


def _greedy_construct_fast(n_rows, k):
    """Faster greedy: try multiple random starting offsets to find compact rows."""
    best_rows = None
    best_max = float('inf')

    for attempt in range(20):
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                found = False
                # Try a small random window first, then expand
                search_limit = current_max + 500
                for x in range(current_max + 1, search_limit):
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break

                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        current_max = x
                        found = True
                        break

                if not found:
                    # Expand search
                    for x in range(search_limit, current_max + 3000):
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break

                        if valid:
                            row.append(x)
                            for a in row[:-1]:
                                used_diffs.add(x - a)
                            current_max = x
                            found = True
                            break

                if not found:
                    success = False
                    break

            if not success:
                break

            rows.append(row)

        if success and rows:
            curr_max = max(max(r) for r in rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = rows

    if best_rows:
        return best_rows
    return _greedy_construct(n_rows, k)
