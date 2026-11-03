"""Difference Triangle Set (4,7) solver.

Structural change: model the problem as selecting n_rows mutually-disjoint
difference-sets (Sidon-like row patterns) within a target scope M.  We use a
greedy DSATUR-inspired coloring / packing approach:
  1. For a candidate scope M, enumerate many valid row patterns (each a set of
     k+1 numbers starting at 0 with distinct internal differences).
  2. Build a conflict graph where edges connect patterns that share any
     difference value.
  3. Greedily select n_rows independent patterns (no shared differences) using
     a DSATUR-style heuristic (prefer patterns with fewest conflicts with
     already-chosen patterns, tie-break by smallest max entry).
  4. Binary search on M (or linear scan) to find the smallest M where n_rows
     disjoint patterns exist.
  5. Polish the result with local compaction and limited simulated annealing.

This is a fundamentally different representation: instead of constructing rows
sequentially and compacting, we treat each valid row as a node in a graph and
find an independent set of size n_rows.
"""
import time
import random
import math


def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    L = len(row)
    diffs = set()
    for i in range(L):
        for j in range(i + 1, L):
            diffs.add(row[j] - row[i])
    return diffs


def _generate_patterns(k, M, rng, n_patterns=2000):
    """Generate up to n_patterns valid row patterns with max entry <= M.

    A pattern is a list of k+1 increasing integers starting at 0.
    Validity: all internal differences are distinct (Sidon property within row).
    """
    patterns = []
    L = k + 1
    # Use randomized greedy to generate diverse patterns
    for _ in range(n_patterns):
        row = [0]
        x = 0
        row_diffs = set()
        valid = True
        while len(row) < L:
            x += 1
            if x > M:
                valid = False
                break
            # Randomized step
            if rng.random() < 0.15:
                x += rng.randint(1, 3)
                if x > M:
                    valid = False
                    break
            new_diffs = {x - a for a in row}
            if new_diffs & row_diffs:
                continue
            row.append(x)
            row_diffs |= new_diffs
        if valid and len(row) == L:
            # Check internal Sidon property more carefully
            all_d = set()
            ok = True
            for i in range(L):
                for j in range(i + 1, L):
                    d = row[j] - row[i]
                    if d in all_d:
                        ok = False
                        break
                    all_d.add(d)
                if not ok:
                    break
            if ok:
                patterns.append(row)
    return patterns


def _generate_patterns_greedy(k, M, rng, n_patterns=3000):
    """Generate patterns using multiple greedy strategies for diversity."""
    patterns = []
    L = k + 1
    for attempt in range(n_patterns):
        strategy = attempt % 4
        row = [0]
        x = 0
        row_diffs = set()
        valid = True
        while len(row) < L:
            if strategy == 0:
                x += 1
            elif strategy == 1:
                x += rng.randint(1, 2)
            elif strategy == 2:
                x += 1
                if rng.random() < 0.3:
                    x += 1
            else:
                x += rng.randint(1, 3)
            
            if x > M:
                valid = False
                break
            new_diffs = {x - a for a in row}
            if new_diffs & row_diffs:
                continue
            row.append(x)
            row_diffs |= new_diffs
        if valid and len(row) == L:
            all_d = set()
            ok = True
            for i in range(L):
                for j in range(i + 1, L):
                    d = row[j] - row[i]
                    if d in all_d:
                        ok = False
                        break
                    all_d.add(d)
                if not ok:
                    break
            if ok:
                patterns.append(row)
    return patterns


def _dsatur_select(patterns, n_rows, k, rng):
    """Select n_rows mutually-disjoint patterns using DSATUR-inspired heuristic.

    Returns the selected rows or None if not enough can be found.
    """
    if len(patterns) < n_rows:
        return None
    
    # Precompute difference sets
    pat_diffs = [_row_diffs(p) for p in patterns]
    pat_maxs = [p[-1] for p in patterns]
    
    selected = []
    selected_diffs = set()
    available = list(range(len(patterns)))
    
    for _ in range(n_rows):
        if not available:
            return None
        # Score each available pattern:
        # Primary: number of conflicts with already-selected patterns (fewer is better)
        # Secondary: max entry (smaller is better)
        # Tertiary: random tiebreaker
        best_idx = None
        best_score = None
        for idx in available:
            conflicts = len(pat_diffs[idx] & selected_diffs)
            if conflicts > 0:
                # This pattern shares differences with selected ones - skip
                continue
            score = (conflicts, pat_maxs[idx], rng.random())
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is None:
            # No conflict-free pattern available; try with minimal conflicts
            best_idx = None
            best_conflicts = float('inf')
            best_max = float('inf')
            for idx in available:
                conflicts = len(pat_diffs[idx] & selected_diffs)
                if conflicts < best_conflicts or (conflicts == best_conflicts and pat_maxs[idx] < best_max):
                    best_conflicts = conflicts
                    best_max = pat_maxs[idx]
                    best_idx = idx
            if best_idx is None or best_conflicts > 0:
                return None
        
        selected.append(patterns[best_idx])
        selected_diffs |= pat_diffs[best_idx]
        available.remove(best_idx)
    
    return selected


def _dsatur_select_v2(patterns, n_rows, k, rng):
    """Alternative selection: sort by max entry, then greedily pick non-conflicting."""
    if len(patterns) < n_rows:
        return None
    
    pat_diffs = [_row_diffs(p) for p in patterns]
    
    # Sort by max entry
    order = sorted(range(len(patterns)), key=lambda i: (patterns[i][-1], rng.random()))
    
    selected = []
    selected_diffs = set()
    
    for idx in order:
        if len(selected) == n_rows:
            break
        if pat_diffs[idx] & selected_diffs:
            continue
        selected.append(patterns[idx])
        selected_diffs |= pat_diffs[idx]
    
    if len(selected) == n_rows:
        return selected
    return None


def _local_compact(rows, n_rows, k, time_limit):
    """Iteratively compact each element to its smallest feasible value."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = set()
    for row in work:
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(row[j] - row[i])
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for r_idx in range(n_rows):
            row = work[r_idx]
            for pos in range(L - 1, 0, -1):
                old_val = row[pos]
                lower = row[pos - 1] + 1
                if lower >= old_val:
                    continue
                old_diffs = set()
                for i in range(L):
                    if i < pos:
                        old_diffs.add(row[pos] - row[i])
                    elif i > pos:
                        old_diffs.add(row[i] - row[pos])
                for new_val in range(lower, old_val):
                    if pos < L - 1 and new_val >= row[pos + 1]:
                        break
                    new_diffs = set()
                    for i in range(L):
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        elif i > pos:
                            new_diffs.add(row[i] - new_val)
                    other = used_set - old_diffs
                    if new_diffs & other:
                        continue
                    for d in old_diffs:
                        used_set.discard(d)
                    for d in new_diffs:
                        used_set.add(d)
                    row[pos] = new_val
                    improved = True
                    break
    return work


def _can_set(row, pos, new_val, used_set, L):
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < L - 1 and new_val >= row[pos + 1]:
        return False
    old_diffs = set()
    for i in range(L):
        if i < pos:
            old_diffs.add(row[pos] - row[i])
        elif i > pos:
            old_diffs.add(row[i] - row[pos])
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    other = used_set - old_diffs
    if new_diffs & other:
        return False
    return True


def _apply_set(row, pos, new_val, used_set, L):
    old_diffs = set()
    for i in range(L):
        if i < pos:
            old_diffs.add(row[pos] - row[i])
        elif i > pos:
            old_diffs.add(row[i] - row[pos])
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    for d in old_diffs:
        used_set.discard(d)
    for d in new_diffs:
        used_set.add(d)
    row[pos] = new_val


def _fast_compact_row(row, used_set, L):
    """Compact a single row in-place."""
    for pos in range(L - 1, 0, -1):
        old_val = row[pos]
        lower = row[pos - 1] + 1
        if lower >= old_val:
            continue
        old_diffs = set()
        for i in range(L):
            if i < pos:
                old_diffs.add(row[pos] - row[i])
            elif i > pos:
                old_diffs.add(row[i] - row[pos])
        for new_val in range(lower, old_val):
            if pos < L - 1 and new_val >= row[pos + 1]:
                break
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            other = used_set - old_diffs
            if new_diffs & other:
                continue
            for d in old_diffs:
                used_set.discard(d)
            for d in new_diffs:
                used_set.add(d)
            row[pos] = new_val
            break


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = set()
    for row in work:
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(row[j] - row[i])
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max

    start_temp = max(1, current_max // 6)
    end_temp = 0.05
    cooling = 0.9997
    t = start_temp
    stagnation_counter = 0
    max_stagnation = 2000

    while time.time() < time_limit and t > end_temp:
        move_type = rng.random()

        if move_type < 0.35:
            # Decrease a random element
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower < old_val:
                new_val = rng.randint(lower, old_val - 1)
                if new_val < old_val and _can_set(row, pos, new_val, used_set, L):
                    _apply_set(row, pos, new_val, used_set, L)
                    _fast_compact_row(row, used_set, L)
                    new_max = _max_entry(work)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in work]
                        current_max = best_max
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

        elif move_type < 0.60:
            # Increase a random element (perturbation), then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(3, current_max // 4))
            new_val = rng.randint(old_val + 1, upper_limit)
            if new_val > old_val and _can_set(row, pos, new_val, used_set, L):
                _apply_set(row, pos, new_val, used_set, L)
                _fast_compact_row(row, used_set, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

        elif move_type < 0.75:
            # Full recompact
            for r_idx in range(n_rows):
                _fast_compact_row(work[r_idx], used_set, L)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        else:
            # Swap elements between rows
            r1 = rng.randint(0, n_rows - 1)
            r2 = rng.randint(0, n_rows - 1)
            if r1 != r2:
                p1 = rng.randint(1, L - 1)
                p2 = rng.randint(1, L - 1)
                row1 = work[r1]
                row2 = work[r2]
                v1 = row1[p1]
                v2 = row2[p2]
                
                bounds_ok = True
                if p1 > 0 and v2 <= row1[p1-1]:
                    bounds_ok = False
                if p1 < L-1 and v2 >= row1[p1+1]:
                    bounds_ok = False
                if p2 > 0 and v1 <= row2[p2-1]:
                    bounds_ok = False
                if p2 < L-1 and v1 >= row2[p2+1]:
                    bounds_ok = False
                
                if bounds_ok:
                    old_diffs1 = set()
                    for i in range(L):
                        if i < p1:
                            old_diffs1.add(row1[p1] - row1[i])
                        elif i > p1:
                            old_diffs1.add(row1[i] - row1[p1])
                    old_diffs2 = set()
                    for i in range(L):
                        if i < p2:
                            old_diffs2.add(row2[p2] - row2[i])
                        elif i > p2:
                            old_diffs2.add(row2[i] - row2[p2])
                    
                    new_diffs1 = set()
                    for i in range(L):
                        if i < p1:
                            new_diffs1.add(v2 - row1[i])
                        elif i > p1:
                            new_diffs1.add(row1[i] - v2)
                    
                    new_diffs2 = set()
                    for i in range(L):
                        if i < p2:
                            new_diffs2.add(v1 - row2[i])
                        elif i > p2:
                            new_diffs2.add(row2[i] - v1)
                    
                    other = used_set - old_diffs1 - old_diffs2
                    if not (new_diffs1 & other) and not (new_diffs2 & other) and not (new_diffs1 & new_diffs2):
                        for d in old_diffs1:
                            used_set.discard(d)
                        for d in old_diffs2:
                            used_set.discard(d)
                        for d in new_diffs1:
                            used_set.add(d)
                        for d in new_diffs2:
                            used_set.add(d)
                        row1[p1] = v2
                        row2[p2] = v1
                        _fast_compact_row(row1, used_set, L)
                        _fast_compact_row(row2, used_set, L)
                        new_max = _max_entry(work)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(r) for r in work]
                            current_max = best_max
                            stagnation_counter = 0
                        else:
                            stagnation_counter += 1
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

        if stagnation_counter > max_stagnation:
            # Restart with fresh pattern-based construction
            fresh = _pattern_based_construct(n_rows, k, rng, time.time() + 5)
            if fresh is not None and _check_valid(fresh, k):
                fresh_max = _max_entry(fresh)
                if fresh_max < best_max:
                    best_max = fresh_max
                    best_rows = [list(r) for r in fresh]
                    current_max = best_max
                    work = [list(r) for r in fresh]
                    used_set = set()
                    for row in work:
                        for i in range(L):
                            for j in range(i + 1, L):
                                used_set.add(row[j] - row[i])
                    t = start_temp
                    stagnation_counter = 0

        t *= cooling

    return best_rows


def _pattern_based_construct(n_rows, k, rng, deadline):
    """Construct rows using pattern-based approach with a time deadline."""
    L = k + 1
    # Try increasing scopes
    for M in range(100, 160):
        if time.time() > deadline:
            return None
        patterns = _generate_patterns_greedy(k, M, rng, n_patterns=1500)
        if len(patterns) < n_rows:
            continue
        # Try DSATUR selection
        result = _dsatur_select(patterns, n_rows, k, rng)
        if result is not None:
            return result
        result = _dsatur_select_v2(patterns, n_rows, k, rng)
        if result is not None:
            return result
    return None


def _greedy_construct(n_rows, k, rng=None):
    """Simple greedy baseline."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 142  # Safety margin before 150s kill

    # 1) Greedy baseline
    rng = random.Random(42)
    best_rows = _greedy_construct(n_rows, k, rng=rng)
    best_max = _max_entry(best_rows)

    # 2) Pattern-based construction (the structural change)
    # Try to find a better solution via the graph/coloring approach
    pattern_deadline = time_limit - 50
    for M in range(110, 145):
        if time.time() > pattern_deadline:
            break
        patterns = _generate_patterns_greedy(k, M, rng, n_patterns=2000)
        if len(patterns) < n_rows:
            continue
        result = _dsatur_select(patterns, n_rows, k, rng)
        if result is not None:
            rmax = _max_entry(result)
            if rmax < best_max:
                best_rows = result
                best_max = rmax
            break  # Found valid at this M, no need to try larger M
        result = _dsatur_select_v2(patterns, n_rows, k, rng)
        if result is not None:
            rmax = _max_entry(result)
            if rmax < best_max:
                best_rows = result
                best_max = rmax
            break

    # 3) Local compaction
    compact_deadline = time_limit - 20
    compacted = _local_compact(best_rows, n_rows, k, compact_deadline)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax

    # 4) Simulated annealing with remaining time
    if time.time() < time_limit - 2:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 2)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 5) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
